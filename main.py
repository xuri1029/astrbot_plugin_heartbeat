"""心念一动（astrbot_plugin_heartbeat）

每隔一段随机时间自己醒一次：看一眼现在是什么时候、上次聊天过去多久、
有没有没做完的事，然后决定——说点什么、发条说说、记一笔，还是什么都不做。

设计要点：
1. 不碰聊天主链（不改 system_prompt、不拦消息），只做独立的一次思考。
2. 工具走 AstrBot 全局工具表，因此理论上能调用其它插件暴露的能力。
3. 默认不删任何数据，黑名单里的工具永远不碰。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, StarTools

PLUGIN_NAME = "astrbot_plugin_heartbeat"

THINK_SYSTEM_PROMPT = """你不是在回复谁，你是在自己的时间里。

你会收到一份“现状”，然后决定这一段时间要不要做点什么。你可以：
- 用工具去查、去想、去准备（工具就是你平时能用的一切能力）
- 也可以决定什么都不做，这完全允许，而且很多时候是对的

硬规则：
1. 不要为了说话而说话。刚聊过、对方在忙、没什么值得说的，就不做。
2. 不要重复上一次已经做过的同类动作，除非确实有新东西。
3. 安静时段不做任何会发出声音或动静的事。
4. 想清楚再动。宁可多想几轮，也不要随口一句没意思的话。

最后必须用一段 JSON 收尾，格式：
{"action": "message|feed|note|none", "text": "要发的内容（没有就空字符串）", "reason": "你为什么这么决定，一句话"}
"""


class HeartbeatPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self.data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "state.json"
        self.log_path = self.data_dir / "wake_log.jsonl"
        self.state: dict[str, Any] = self._load_state()
        self._task: asyncio.Task | None = None
        self._running = False
        self._busy = False
        self._last_prompt_tokens = 0

    # ---------------------------------------------------------------- 配置读取
    def _cfg(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _owner_umo(self) -> str:
        return self._resolve_owner_umo()

    def _quiet_window(self) -> tuple[int, int]:
        def parse(text: str, fallback: int) -> int:
            try:
                hour, minute = str(text).strip().split(":")
                return int(hour) * 60 + int(minute)
            except Exception:
                return fallback

        return (
            parse(self._cfg("quiet_start", "01:00"), 60),
            parse(self._cfg("quiet_end", "09:00"), 540),
        )

    def _in_quiet(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        start, end = self._quiet_window()
        current = now.hour * 60 + now.minute
        if start == end:
            return False
        if start < end:
            return start <= current < end
        return current >= start or current < end

    # ---------------------------------------------------------------- 状态存取
    def _load_state(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[心念一动] 状态读取失败: %s", exc)
            return {}

    def _save_state(self) -> None:
        try:
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.state_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[心念一动] 状态写入失败: %s", exc)

    def _append_log(self, entry: dict[str, Any]) -> None:
        entry = {"at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **entry}
        try:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            keep = max(20, int(self._cfg("log_keep", 300)))
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > keep:
                self.log_path.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[心念一动] 日志写入失败: %s", exc)

    # ---------------------------------------------------------------- 生命周期
    async def initialize(self) -> None:  # pragma: no cover - 由框架调用
        self._ensure_task()

    @filter.on_astrbot_loaded()
    async def _on_loaded(self):
        self._ensure_task()

    def _ensure_task(self) -> None:
        if self._task and not self._task.done():
            return
        if not bool(self._cfg("enabled", True)):
            logger.info("[心念一动] 未启用，不启动循环")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="heartbeat-loop")
        logger.info("[心念一动] 已启动，先睡 %.0f 分钟后第一次醒来", self._next_gap_minutes())

    async def terminate(self) -> None:  # pragma: no cover
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- 活动记录
    @filter.on_llm_request()
    async def _track_activity(self, event: AstrMessageEvent, req: Any = None):
        try:
            umo = getattr(event, "unified_msg_origin", "") or ""
            text = str(getattr(event, "message_str", "") or "").strip()
            raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
            # 只有真实平台消息（带原始报文）才算“刚聊过”；
            # 插件内部伪造的事件 raw_message 为 None，不能当成人说话。
            if umo and text and raw is not None and not str(self._cfg("owner_umo", "") or "").strip():
                if not str(self.state.get("owner_umo") or "").strip():
                    self.state["owner_umo"] = umo
                    logger.info("[心念一动] 已记住主人会话: %s", umo)
            if umo == self._owner_umo() and text and raw is not None:
                self.state["last_user_at"] = time.time()
                self.state["last_user_text"] = text[:80]
                self._save_state()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- 主循环
    def _next_gap_minutes(self) -> float:
        low = max(1.0, float(self._cfg("interval_min_minutes", 30)))
        high = max(low, float(self._cfg("interval_max_minutes", 80)))
        return random.uniform(low, high)

    async def _loop(self) -> None:
        await asyncio.sleep(20)
        while self._running:
            gap = self._next_gap_minutes()
            self.state["next_wake_at"] = time.time() + gap * 60
            self._save_state()
            logger.info("[心念一动] 下次醒来: %s (%.0f 分钟后)",
                        datetime.fromtimestamp(self.state["next_wake_at"]).strftime("%H:%M"), gap)
            try:
                await asyncio.sleep(gap * 60)
            except asyncio.CancelledError:
                raise
            if not self._running:
                return
            if bool(self._cfg("enabled", True)):
                await self._wake("timer")

    # ---------------------------------------------------------------- 醒来
    async def _wake(self, reason: str) -> dict[str, Any]:
        if self._busy:
            return {"skipped": "busy"}
        if not self._owner_umo():
            logger.info("[心念一动] 还不知道对谁说话，先跳过（等你发一条私聊）")
            return {"skipped": "no_owner"}
        if self._in_quiet():
            logger.info("[心念一动] 安静时段，跳过")
            return {"skipped": "quiet"}
        self._busy = True
        try:
            snapshot, flags = await self._build_snapshot()
            result = await self._think(snapshot)
            action = result.get("action", "none")
            text = str(result.get("text") or "").strip()
            allowed = await self._apply_action(action, text, flags, reason)
            entry = {
                "reason": reason,
                "action": action,
                "allowed": allowed,
                "text": text[:300],
                "think": str(result.get("reason") or "")[:300],
                "rounds": result.get("rounds", 0),
                "tools": result.get("tools", []),
                "tokens": result.get("tokens", 0),
                "skipped": flags,
            }
            self._append_log(entry)
            self.state["last_wake_at"] = time.time()
            self._save_state()
            logger.info("[心念一动] 醒来结束 action=%s allowed=%s 说明=%s",
                        action, allowed, entry["think"][:80])
            return entry
        except Exception as exc:  # noqa: BLE001
            logger.error("[心念一动] 思考失败: %s", exc, exc_info=True)
            self._append_log({"reason": reason, "error": str(exc)[:300]})
            return {"error": str(exc)}
        finally:
            self._busy = False

    # ---------------------------------------------------------------- 现状快照
    async def _build_snapshot(self) -> tuple[str, dict[str, Any]]:
        now = datetime.now()
        gap_text = "不知道"
        last_user_at = float(self.state.get("last_user_at") or 0)
        if last_user_at:
            minutes = max(0, int((time.time() - last_user_at) / 60))
            gap_text = f"{minutes} 分钟前" if minutes < 120 else f"{minutes // 60} 小时前"

        parts = [
            f"现在是 {now.strftime('%Y-%m-%d %H:%M')}（{'凌晨' if now.hour < 6 else '上午' if now.hour < 12 else '下午' if now.hour < 18 else '晚上'}），星期{'一二三四五六日'[now.weekday()]}。",
            f"主人最后说话：{gap_text}。",
        ]

        pending = self._read_pending_intents()
        if pending:
            parts.append("你之前答应过、但还没做完的事：" + "；".join(pending[:5]))

        recent = self._recent_decisions(3)
        if recent:
            parts.append("最近几次你自己决定过什么：" + "；".join(recent))

        polite = self._recent_proactive_within()
        flags = {
            "last_user_at": last_user_at,
            "user_gap_minutes": int((time.time() - last_user_at) / 60) if last_user_at else 10**6,
            "polite_blocked": polite,
        }
        if polite:
            parts.append("注意：最近几小时内已经有主动消息发出去了，这一轮你只适合想，不适合再说。")

        parts.append("如果你此刻有想做的，就动手；没有就明说没有。")
        return "\n".join(parts), flags

    def _read_pending_intents(self) -> list[str]:
        """读取念念不忘（若安装）里尚未完成的事。"""
        candidates = [
            Path(StarTools.get_data_dir("astrbot_plugin_intent_keeper")) / "intents.json",
        ]
        for path in candidates:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items = raw.get(self._owner_umo()) if isinstance(raw, dict) else None
            if not isinstance(items, list):
                continue
            out = []
            for item in items:
                if isinstance(item, dict) and item.get("status") == "pending":
                    topic = str(item.get("topic") or "").strip()
                    if topic:
                        out.append(topic[:40])
            if out:
                return out
        return []

    def _recent_decisions(self, limit: int) -> list[str]:
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()[-limit:]
        except FileNotFoundError:
            return []
        except Exception:
            return []
        out = []
        for line in lines:
            try:
                item = json.loads(line)
            except Exception:
                continue
            action = item.get("action") or item.get("error") or ""
            text = item.get("text") or item.get("think") or ""
            if action:
                out.append(f"{item.get('at', '')[-5:]} {action}：{str(text)[:40]}")
        return out

    def _recent_proactive_within(self) -> bool:
        hours = float(self._cfg("polite_hours", 3))
        if hours <= 0:
            return False
        last = float(self.state.get("last_sent_at") or 0)
        return bool(last) and (time.time() - last) < hours * 3600

    # ---------------------------------------------------------------- 思考
    def _available_tools(self) -> Any:
        try:
            manager = self.context.get_llm_tool_manager()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[心念一动] 拿不到工具表: %s", exc)
            return None
        from astrbot.core.agent.tool import ToolSet

        blacklist = {str(x) for x in (self._cfg("tool_blacklist", []) or [])}
        mode = str(self._cfg("tools_mode", "all")).lower()
        whitelist = {str(x) for x in (self._cfg("tool_whitelist", []) or [])}
        tools = []
        for tool in list(getattr(manager, "func_list", []) or []):
            name = str(getattr(tool, "name", "") or "")
            if not name or name in blacklist:
                continue
            if not bool(getattr(tool, "active", True)):
                continue
            if mode == "whitelist" and name not in whitelist:
                continue
            tools.append(tool)
        return ToolSet(tools=tools) if tools else None

    async def _think(self, snapshot: str) -> dict[str, Any]:
        provider = self._resolve_provider()
        if provider is None:
            return {"action": "none", "reason": "没有可用模型"}

        toolset = self._available_tools()
        contexts: list[dict[str, Any]] = [{"role": "user", "content": snapshot}]
        deadline = time.monotonic() + float(self._cfg("max_think_seconds", 120))
        max_rounds = max(1, int(self._cfg("max_rounds", 8)))
        used_tools: list[str] = []
        tokens = 0
        last_text = ""

        for round_index in range(max_rounds):
            if time.monotonic() > deadline:
                break
            try:
                resp = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=None,
                        session_id=self._owner_umo(),
                        contexts=contexts,
                        system_prompt=THINK_SYSTEM_PROMPT,
                        func_tool=toolset,
                    ),
                    timeout=max(5.0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                logger.warning("[心念一动] 第 %d 轮思考超时", round_index + 1)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("[心念一动] 第 %d 轮调用失败: %s", round_index + 1, exc)
                break

            text = str(getattr(resp, "completion_text", "") or "").strip()
            if text:
                last_text = text
            try:
                usage = getattr(resp, "usage", None)
                if isinstance(usage, dict):
                    tokens += int(usage.get("total_tokens") or 0)
                elif usage is not None:
                    tokens += int(getattr(usage, "total_tokens", 0) or 0)
            except Exception:
                pass

            names = list(getattr(resp, "tools_call_name", []) or [])
            args_list = list(getattr(resp, "tools_call_args", []) or [])
            if not names:
                return self._parse_decision(last_text, round_index + 1, used_tools, tokens)

            for idx, name in enumerate(names):
                raw_args = args_list[idx] if idx < len(args_list) else {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args) if raw_args.strip() else {}
                    except Exception:
                        raw_args = {}
                if not isinstance(raw_args, dict):
                    raw_args = {}
                used_tools.append(str(name))
                logger.info("[心念一动] 调工具 %s(%s)", name, json.dumps(raw_args, ensure_ascii=False)[:120])
                output = await self._exec_tool(str(name), raw_args)
                contexts.append(
                    {
                        "role": "user",
                        "content": f"[工具 {name} 的结果]\n{str(output)[:1500]}",
                    }
                )

        return self._parse_decision(last_text, max_rounds, used_tools, tokens)

    def _parse_decision(self, text: str, rounds: int, tools: list[str], tokens: int) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if text:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = {}
        if not isinstance(data, dict):
            data = {}
        action = str(data.get("action") or "none").strip().lower()
        if action not in {"message", "feed", "note", "none"}:
            action = "none"
        return {
            "action": action,
            "text": str(data.get("text") or ""),
            "reason": str(data.get("reason") or text or ""),
            "rounds": rounds,
            "tools": tools,
            "tokens": tokens,
        }

    def _resolve_provider(self) -> Any:
        provider_id = str(self._cfg("provider_id", "") or "").strip()
        try:
            if provider_id:
                return self.context.get_provider_by_id(provider_id)
        except Exception:  # noqa: BLE001
            pass
        for kwargs in ({"umo": self._owner_umo()}, {}):
            try:
                provider = self.context.get_using_provider(**kwargs)
                if provider is not None:
                    return provider
            except Exception:  # noqa: BLE001
                continue
        return None

    # ---------------------------------------------------------------- 执行动作
    async def _apply_action(
        self, action: str, text: str, flags: dict[str, Any], reason: str
    ) -> bool:
        if action == "none":
            return False
        if flags.get("user_gap_minutes", 10**6) < int(self._cfg("min_gap_after_user_minutes", 10)):
            logger.info("[心念一动] 刚聊过，这一轮不打扰")
            return False
        if flags.get("polite_blocked") and action == "message":
            logger.info("[心念一动] 最近已有主动消息，这轮让路")
            return False
        if action == "message":
            if not bool(self._cfg("allow_send_message", True)) or not text:
                return False
            ok = await self._send(text)
            if ok:
                self.state["last_sent_at"] = time.time()
                self._save_state()
            return ok
        if action == "feed":
            if not bool(self._cfg("allow_publish_feed", True)) or not text:
                return False
            output = await self._exec_tool("pc_qzone_publish_feed", {"content": text})
            ok = "失败" not in output and "error" not in output.lower()
            if ok:
                self.state["last_sent_at"] = time.time()
                self._save_state()
            return ok
        if action == "note":
            if not bool(self._cfg("allow_note", True)) or not text:
                return False
            try:
                path = self.data_dir / "notes.jsonl"
                with path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps({"at": datetime.now().isoformat(), "text": text}, ensure_ascii=False) + "\n")
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("[心念一动] 记笔记失败: %s", exc)
                return False
        return False

    async def _send(self, text: str) -> bool:
        try:
            chain = MessageChain(chain=[Plain(text=text)])
            await self.context.send_message(self._owner_umo(), chain)
            logger.info("[心念一动] 已发出: %s", text[:60])
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[心念一动] 发送失败: %s", exc)
            return False

    def _build_event(self) -> Any:
        try:
            from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
            from astrbot.core.platform.message_type import MessageType

            platforms = []
            try:
                platforms = list(self.context.platform_manager.get_insts())
            except Exception:
                platforms = []
            if not platforms:
                return None
            platform = platforms[0]
            message_obj = AstrBotMessage()
            message_obj.type = MessageType.FRIEND_MESSAGE
            message_obj.self_id = self._self_id(platform)
            message_obj.session_id = self._owner_umo()
            message_obj.message_id = f"heartbeat_{int(time.time())}"
            message_obj.sender = MessageMember(user_id=self._owner_user_id(), nickname="")
            message_obj.message = [Plain(text="")]
            message_obj.message_str = ""
            message_obj.raw_message = None
            message_obj.timestamp = int(time.time())
            return AstrMessageEvent("", message_obj, platform.meta(), message_obj.session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[心念一动] 构造事件失败: %s", exc)
            return None

    async def _exec_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            manager = self.context.get_llm_tool_manager()
            tool = manager.get_func(name)
        except Exception as exc:  # noqa: BLE001
            return f"工具表不可用: {exc}"
        if tool is None:
            return f"没有这个工具: {name}"
        handler = getattr(tool, "handler", None)
        if handler is None:
            return f"工具 {name} 没有可直接调用的处理器"
        event = self._build_event()
        if event is None:
            return "无法构造消息事件，工具未能执行"
        try:
            result = handler(event, **args)
            if inspect.isasyncgen(result):
                chunks: list[str] = []
                async for item in result:
                    if item is None:
                        continue
                    chunks.append(str(item))
                return "\n".join(chunks) or "已执行"
            if inspect.isawaitable(result):
                result = await result
            return str(result) if result is not None else "已执行"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[心念一动] 工具 %s 执行失败: %s", name, exc)
            return f"执行失败: {exc}"

    def _owner_user_id(self) -> str:
        umo = self._owner_umo()
        return umo.split(":")[-1] if umo else ""

    def _self_id(self, platform: Any) -> str:
        for getter in ("get_self_id", "self_id"):
            try:
                value = getattr(platform, getter, None)
                value = value() if callable(value) else value
                if value:
                    return str(value)
            except Exception:  # noqa: BLE001
                continue
        try:
            return str(getattr(platform.meta(), "id", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _resolve_owner_umo(self) -> str:
        """优先读配置；没配就用记住的会话，都没有则返回空。"""
        configured = str(self._cfg("owner_umo", "") or "").strip()
        if configured:
            return configured
        return str(self.state.get("owner_umo") or "").strip()
    # ---------------------------------------------------------------- 指令
    @filter.command("心跳")
    async def cmd_heartbeat(self, event: AstrMessageEvent):
        text = str(getattr(event, "message_str", "") or "").replace("/心跳", "").strip()
        sub = text.split()[0] if text else "状态"
        if sub in {"状态", "status"}:
            next_at = self.state.get("next_wake_at")
            next_text = (
                datetime.fromtimestamp(next_at).strftime("%H:%M") if next_at else "未安排"
            )
            last_user = self.state.get("last_user_at")
            last_user_text = (
                datetime.fromtimestamp(last_user).strftime("%m-%d %H:%M") if last_user else "从没记到"
            )
            yield event.plain_result(
                "心念一动\n"
                f"状态：{'运行中' if self._running else '停止'}\n"
                f"下次醒来：{next_text}\n"
                f"上次主人说话：{last_user_text}\n"
                f"安静时段：{self._cfg('quiet_start', '01:00')} - {self._cfg('quiet_end', '09:00')}\n"
                f"间隔：{self._cfg('interval_min_minutes', 30)}-{self._cfg('interval_max_minutes', 80)} 分钟"
            )
            return
        if sub in {"立即", "now"}:
            yield event.plain_result("好，我现在醒一次，想完了告诉你。")
            entry = await self._wake("manual")
            yield event.plain_result(
                "想完了：\n"
                f"动作：{entry.get('action', entry.get('error', '无'))}\n"
                f"轮数：{entry.get('rounds', 0)}　工具：{', '.join(entry.get('tools', []) or []) or '没用'}\n"
                f"理由：{entry.get('think', '')[:200]}\n"
                f"内容：{entry.get('text', '')[:300]}"
            )
            return
        if sub in {"日志", "log"}:
            try:
                lines = self.log_path.read_text(encoding="utf-8").splitlines()[-5:]
            except Exception:
                lines = []
            if not lines:
                yield event.plain_result("还没有记录。")
                return
            out = []
            for line in lines:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                out.append(
                    f"{item.get('at', '')[5:16]} {item.get('action', item.get('error', '?'))}："
                    f"{str(item.get('think', ''))[:60]}"
                )
            yield event.plain_result("最近几次：\n" + "\n".join(out))
            return
        if sub in {"暂停", "stop"}:
            self._running = False
            if self._task and not self._task.done():
                self._task.cancel()
            yield event.plain_result("已暂停，不会再自己醒来。")
            return
        if sub in {"继续", "start"}:
            self._ensure_task()
            yield event.plain_result("已恢复。")
            return
        yield event.plain_result("用法：/心跳 状态 | 立即 | 日志 | 暂停 | 继续")
