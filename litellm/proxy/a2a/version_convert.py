# pyright: reportUnknownArgumentType=false
# a2a-sdk's compat conversions (pb2_v10, ParseDict, MessageToDict, to_compat_*)
# are protobuf-generated/untyped, so every conversion call here takes Unknown-typed
# arguments. This module is the A2A 0.3<->1.0 boundary; the rule is off file-wide
# rather than scattering per-line ignores across every SDK call.
"""
Normalize A2A JSON-RPC payloads to the protocol version LiteLLM serves for an agent.

LiteLLM fronts upstream agents and lets an admin pin the protocol version it speaks
to clients (``0.3`` or ``1.0``) per agent. Upstream responses may arrive in either
wire shape, so every response, stream event, forwarded request and extended card is
converted to the served version here. Conversion is shape-detecting (we infer the
payload's current version rather than trusting a stored one) and best-effort: any
failure falls back to returning the input unchanged so a conversion bug can never
break an otherwise-valid response.

The two wire shapes:

- ``0.3``: JSON dump of the compat pydantic types, discriminated by a ``kind`` field
  (``message`` / ``task`` / ``status-update`` / ``artifact-update``). A send result is
  the bare object.
- ``1.0``: protobuf JSON (``MessageToDict``), a oneof envelope keyed by
  ``message`` / ``task`` / ``statusUpdate`` / ``artifactUpdate`` with no ``kind``. A
  ``Task`` result is a bare object without ``kind``.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.a2a.agent_card import normalize_protocol_version

A2AVersion = Literal["0.3", "1.0"]
RequestId = str | int | None
JsonDict = dict[str, object]

_V1_SEND_ENVELOPE_KEYS: Final = frozenset({"message", "task"})
_V1_STREAM_ENVELOPE_KEYS: Final = frozenset({"message", "task", "statusUpdate", "artifactUpdate"})
_SECURITY_FIELD_NAMES: Final = frozenset({"security", "securitySchemes", "securityRequirements"})
_V1_SECURITY_SCHEME_KEYS: Final = frozenset(
    {
        "apiKeySecurityScheme",
        "httpAuthSecurityScheme",
        "oauth2SecurityScheme",
        "openIdConnectSecurityScheme",
        "mtlsSecurityScheme",
    }
)
_OBJECT_MAP: Final = TypeAdapter(Mapping[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])


def _dump_03(model: BaseModel) -> JsonDict:
    """Dump a compat (0.3) pydantic model to its camelCase wire dict."""
    return model.model_dump(by_alias=True, exclude_none=True, mode="json")


def _best_effort(convert: Callable[[], JsonDict], fallback: JsonDict, *, label: str) -> JsonDict:
    """Run a conversion, returning ``fallback`` unchanged if it raises."""
    try:
        return convert()
    except Exception as e:  # noqa: BLE001 - best-effort passthrough
        verbose_proxy_logger.debug("A2A %s conversion failed: %s", label, e)
        return fallback


def normalize_jsonrpc_response(content: JsonDict, target: A2AVersion, *, method: str) -> JsonDict:
    """Convert a JSON-RPC response's ``result`` to ``target``.

    Errors and non-dict results pass through untouched.
    """
    if content.get("error") is not None:
        return content
    result: Final = content.get("result")
    if not isinstance(result, dict):
        return content

    converted: Final = _convert_result(result, target, method=method, request_id=_as_request_id(content.get("id")))
    if converted is result:
        return content
    return {**content, "result": converted}


def normalize_stream_event(event: JsonDict, target: A2AVersion, *, request_id: RequestId) -> JsonDict:
    """Convert a single streamed JSON-RPC event's ``result`` to ``target``."""
    if event.get("error") is not None:
        return event
    result: Final = event.get("result")
    if not isinstance(result, dict):
        return event

    converted: Final = _convert_stream_result(result, target, request_id=request_id)
    if converted is result:
        return event
    return {**event, "result": converted}


def normalize_request_params(params: JsonDict, served: A2AVersion, *, method: str) -> JsonDict:
    """Down-convert forwarded request ``params`` from the served version to 0.3.

    Upstream agents in this proxy pivot on 0.3 wire format, so when LiteLLM serves
    1.0 the inbound params must be lowered before forwarding. A no-op when the served
    version is already 0.3.
    """
    if served == "0.3":
        return params
    return _best_effort(
        lambda: _lower_request_params(params, method=method),
        params,
        label=f"request params ({method})",
    )


def _detect_card_version(card: JsonDict) -> A2AVersion:
    """Infer the wire version of an agent card dict.

    ``protocolVersion`` is the authoritative indicator; semver values normalize to
    their major.minor (``"0.3.0"`` -> ``"0.3"``). Fall back to presence of
    ``supportedInterfaces`` (a 1.0-only field) only when the explicit field is
    absent or unrecognized; cards carrying neither signal are treated as 0.3.
    """
    normalized: Final = normalize_protocol_version(card.get("protocolVersion"))
    if normalized is not None:
        return normalized
    return "1.0" if "supportedInterfaces" in card else "0.3"


def _as_map(value: object) -> Mapping[str, object] | None:
    try:
        return _OBJECT_MAP.validate_python(value)
    except ValidationError:
        return None


def _as_list(value: object) -> Sequence[object] | None:
    try:
        return _OBJECT_LIST.validate_python(value)
    except ValidationError:
        return None


def _is_v1_security_scheme(scheme: object) -> bool:
    mapping: Final = _as_map(scheme)
    if mapping is None or len(mapping) != 1:
        return False
    key: Final = next(iter(mapping))
    return key in _V1_SECURITY_SCHEME_KEYS


def _is_flat_security_scheme(scheme: object) -> bool:
    mapping: Final = _as_map(scheme)
    return mapping is not None and "type" in mapping and not _is_v1_security_scheme(scheme)


def _schemes_match_v1(schemes: object) -> bool:
    if schemes is None:
        return True
    mapping: Final = _as_map(schemes)
    if mapping is None:
        return False
    return all(_is_v1_security_scheme(value) for value in mapping.values())


def _schemes_match_flat(schemes: object) -> bool:
    if schemes is None:
        return True
    mapping: Final = _as_map(schemes)
    if mapping is None:
        return False
    return all(_is_flat_security_scheme(value) for value in mapping.values())


def _security_matches_target(card: Mapping[str, object], target: A2AVersion) -> bool:
    schemes: Final = card.get("securitySchemes")
    if target == "1.0":
        return _schemes_match_v1(schemes) and "security" not in card
    return _schemes_match_flat(schemes) and "securityRequirements" not in card


def _string_scopes(value: object) -> tuple[str, ...]:
    items: Final = _as_list(value)
    if items is None:
        return ()
    return tuple(item for item in items if isinstance(item, str))


def _scope_list_value(value: object) -> JsonDict:
    mapping: Final = _as_map(value)
    if mapping is None:
        return {"list": []}
    scopes: Final = _as_list(mapping.get("list"))
    if scopes is None:
        return {"list": []}
    return {"list": [item for item in scopes]}


def _with_explicit_scope_lists(requirement: Mapping[str, object]) -> JsonDict:
    schemes: Final = _as_map(requirement.get("schemes"))
    if schemes is None:
        return {key: value for key, value in requirement.items()}
    filled: Final = {name: _scope_list_value(value) for name, value in schemes.items()}
    return {"schemes": filled}


def _copied_map(value: object) -> JsonDict:
    mapping: Final = _as_map(value)
    if mapping is None:
        return {}
    return {key: item for key, item in mapping.items()}


def _flat_scheme_to_v1(scheme: Mapping[str, object]) -> JsonDict:
    from a2a.compat.v0_3.conversions import MessageToDict, to_core_security_scheme, types_v03

    parsed: Final = types_v03.SecurityScheme.model_validate(dict(scheme))
    return _copied_map(MessageToDict(to_core_security_scheme(parsed), preserving_proto_field_name=False))


def _scheme_to_v1(scheme: object) -> object:
    if _is_v1_security_scheme(scheme):
        return scheme
    mapping: Final = _as_map(scheme)
    if mapping is None:
        return scheme
    try:
        return _flat_scheme_to_v1(mapping)
    except Exception as exc:  # noqa: BLE001 - best-effort passthrough
        verbose_proxy_logger.debug("A2A security scheme conversion failed: %s", exc)
        return scheme


def _schemes_to_v1(schemes: Mapping[str, object]) -> Mapping[str, object]:
    if all(_is_v1_security_scheme(value) for value in schemes.values()):
        return schemes
    return {name: _scheme_to_v1(value) for name, value in schemes.items()}


def _flat_requirement_to_v1(requirement: Mapping[str, object]) -> JsonDict:
    return {"schemes": {name: {"list": list(_string_scopes(raw))} for name, raw in requirement.items()}}


def _requirement_to_v1(item: object) -> object:
    mapping: Final = _as_map(item)
    if mapping is None:
        return item
    if set(mapping) == {"schemes"}:
        return _with_explicit_scope_lists(mapping)
    try:
        return _flat_requirement_to_v1(mapping)
    except Exception as exc:  # noqa: BLE001 - best-effort passthrough
        verbose_proxy_logger.debug("A2A security requirement conversion failed: %s", exc)
        return item


def _requirements_to_v1(security: object) -> Sequence[object] | None:
    items: Final = _as_list(security)
    if items is None:
        return None
    return [_requirement_to_v1(item) for item in items]


def _v1_scheme_to_flat(scheme: Mapping[str, object]) -> JsonDict:
    from a2a.compat.v0_3.conversions import ParseDict, pb2_v10, to_compat_security_scheme

    message: Final = pb2_v10.SecurityScheme()
    ParseDict(dict(scheme), message, ignore_unknown_fields=True)
    dumped: Final = _copied_map(
        to_compat_security_scheme(message).model_dump(  # pyright: ignore[reportAny] # model_dump returns dict[str, Any]
            by_alias=True,
            exclude_none=True,
            mode="json",
        )
    )
    return dumped


def _scheme_to_flat(scheme: object) -> object:
    if not _is_v1_security_scheme(scheme):
        return scheme
    mapping: Final = _as_map(scheme)
    if mapping is None:
        return scheme
    try:
        return _v1_scheme_to_flat(mapping)
    except Exception as exc:  # noqa: BLE001 - best-effort passthrough
        verbose_proxy_logger.debug("A2A security scheme lowering failed: %s", exc)
        return scheme


def _schemes_to_flat(schemes: Mapping[str, object]) -> Mapping[str, object]:
    if all(not _is_v1_security_scheme(value) for value in schemes.values()):
        return schemes
    return {name: _scheme_to_flat(value) for name, value in schemes.items()}


def _v1_requirement_to_flat(requirement: Mapping[str, object]) -> JsonDict:
    schemes: Final = _as_map(requirement.get("schemes"))
    if schemes is None:
        return {}
    return {name: list(_string_scopes(value)) for name, value in schemes.items()}


def _requirement_to_flat(item: object) -> object:
    mapping: Final = _as_map(item)
    if mapping is None or "schemes" not in mapping:
        return item
    try:
        return _v1_requirement_to_flat(mapping)
    except Exception as exc:  # noqa: BLE001 - best-effort passthrough
        verbose_proxy_logger.debug("A2A security requirement lowering failed: %s", exc)
        return item


def _requirements_to_flat(security: object) -> Sequence[object] | None:
    items: Final = _as_list(security)
    if items is None:
        return None
    return [_requirement_to_flat(item) for item in items]


def _v1_requirements_from(card: Mapping[str, object]) -> Sequence[object] | None:
    schemes: Final = card.get("securitySchemes")
    flat: Final = _as_list(card.get("security"))
    existing: Final = _as_list(card.get("securityRequirements"))
    if flat is not None and not _schemes_match_v1(schemes):
        return _requirements_to_v1(flat)
    if existing is not None:
        return _requirements_to_v1(existing)
    return _requirements_to_v1(flat) if flat is not None else None


def _flat_requirements_from(card: Mapping[str, object]) -> Sequence[object] | None:
    schemes: Final = card.get("securitySchemes")
    flat: Final = _as_list(card.get("security"))
    existing: Final = _as_list(card.get("securityRequirements"))
    if flat is not None and not _schemes_match_v1(schemes):
        return flat
    if existing is not None:
        return _requirements_to_flat(existing)
    return flat


def _card_with_security(
    card: Mapping[str, object],
    schemes: object,
    requirements_field: str | None,
    requirements: object,
) -> JsonDict:
    kept: Final = {key: value for key, value in card.items() if key not in _SECURITY_FIELD_NAMES}
    with_schemes: Final = {**kept, "securitySchemes": schemes} if "securitySchemes" in card else kept
    if requirements_field is None:
        return with_schemes
    return {**with_schemes, requirements_field: requirements}


def _card_with_v1_security(card: Mapping[str, object]) -> JsonDict:
    schemes: Final = _as_map(card.get("securitySchemes"))
    requirements: Final = _v1_requirements_from(card)
    converted: Final = _schemes_to_v1(schemes) if schemes is not None else card.get("securitySchemes")
    field: Final = "securityRequirements" if requirements is not None else None
    return _card_with_security(card, converted, field, requirements)


def _card_with_flat_security(card: Mapping[str, object]) -> JsonDict:
    schemes: Final = _as_map(card.get("securitySchemes"))
    requirements: Final = _flat_requirements_from(card)
    converted: Final = _schemes_to_flat(schemes) if schemes is not None else card.get("securitySchemes")
    field: Final = "security" if requirements is not None else None
    return _card_with_security(card, converted, field, requirements)


def align_agent_card_security(card: JsonDict, target: A2AVersion) -> JsonDict:
    """Return ``card`` with security fields in the wire shape ``target`` expects.

    A card whose ``protocolVersion`` is already 1.0 can still carry the flat 0.3
    ``securitySchemes`` / ``security`` fields. Callers that would otherwise skip
    conversion use this so those fields do not leave the proxy unchanged. The same
    object is returned when the fields already match, and the input is not mutated.
    """
    if _security_matches_target(card, target):
        return card
    if target == "1.0":
        return _card_with_v1_security(card)
    return _card_with_flat_security(card)


def normalize_agent_card(card: JsonDict, target: A2AVersion) -> JsonDict:
    """Convert an extended agent card to ``target``.

    When lowering to 0.3, ``additionalInterfaces`` is stripped so the conversion never
    re-exposes upstream backend URLs that the LiteLLM-fronting merge deliberately drops.
    Security fields are aligned to ``target`` even when the rest of the card already
    matches that version.
    """
    if not isinstance(card, dict):
        return card

    current: Final = _detect_card_version(card)
    converted: Final = (
        card
        if current == target and not (target == "0.3" and "supportedInterfaces" in card)
        else _best_effort(lambda: _convert_agent_card(card, target), card, label="agent card")
    )
    return _best_effort(
        lambda: align_agent_card_security(converted, target),
        converted,
        label="agent card security",
    )


def _as_request_id(value: object) -> RequestId:
    return value if isinstance(value, (str, int)) else None


def _convert_result(
    result: JsonDict,
    target: A2AVersion,
    *,
    method: str,
    request_id: RequestId,
) -> JsonDict:
    if method == "message/send":
        return _convert_send_result(result, target, request_id=request_id)
    if method in ("tasks/get", "tasks/cancel"):
        return _convert_task(result, target)
    if method == "tasks/list":
        return _convert_list_tasks_result(result, target)
    return result


def _detect_send_version(result: JsonDict) -> A2AVersion | None:
    if "kind" in result:
        return "0.3"
    if result.keys() & _V1_SEND_ENVELOPE_KEYS:
        return "1.0"
    return None


def _convert_send_result(result: JsonDict, target: A2AVersion, *, request_id: RequestId) -> JsonDict:
    current: Final = _detect_send_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _send_result_to(result, target, request_id),
        result,
        label="send result",
    )


def _send_result_to(result: JsonDict, target: A2AVersion, request_id: RequestId) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_send_message_response,
        to_core_send_message_response,
        types_v03,
    )

    if target == "1.0":
        compat_result: Final = _validate_message_or_task(result)
        response: Final = types_v03.SendMessageResponse(
            root=types_v03.SendMessageSuccessResponse(
                id=str(request_id) if request_id is not None else "",
                result=compat_result,  # pyright: ignore[reportArgumentType]
            )
        )
        return MessageToDict(
            to_core_send_message_response(response),
            preserving_proto_field_name=False,
        )

    pb: Final = pb2_v10.SendMessageResponse()
    ParseDict(result, pb, ignore_unknown_fields=True)
    return _dump_03(to_compat_send_message_response(pb, request_id).root.result)


def _convert_task(result: JsonDict, target: A2AVersion) -> JsonDict:
    current: Final[A2AVersion] = "0.3" if "kind" in result else "1.0"
    if current == target:
        return result
    return _best_effort(lambda: _task_to(result, target), result, label="task")


def _detect_list_tasks_version(result: JsonDict) -> A2AVersion | None:
    tasks: Final = result.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None
    first: Final = tasks[0]
    if not isinstance(first, dict):
        return None
    return "0.3" if "kind" in first else "1.0"


def _convert_list_tasks_result(result: JsonDict, target: A2AVersion) -> JsonDict:
    current: Final = _detect_list_tasks_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _list_tasks_result_to(result, target),
        result,
        label="list tasks result",
    )


def _list_tasks_result_to(result: JsonDict, target: A2AVersion) -> JsonDict:
    tasks: Final = result.get("tasks")
    if not isinstance(tasks, list):
        return result
    return {
        **result,
        "tasks": [_task_to(item, target) if isinstance(item, dict) else item for item in tasks],
    }


def _task_to(result: JsonDict, target: A2AVersion) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_task,
        to_core_task,
        types_v03,
    )

    if target == "1.0":
        core: Final = to_core_task(types_v03.Task.model_validate(result))
        return MessageToDict(core, preserving_proto_field_name=False)

    pb: Final = pb2_v10.Task()
    ParseDict(result, pb, ignore_unknown_fields=True)
    return _dump_03(to_compat_task(pb))


def _detect_stream_version(result: JsonDict) -> A2AVersion | None:
    if "kind" in result:
        return "0.3"
    if result.keys() & _V1_STREAM_ENVELOPE_KEYS:
        return "1.0"
    return None


def _convert_stream_result(result: JsonDict, target: A2AVersion, *, request_id: RequestId) -> JsonDict:
    current: Final = _detect_stream_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _stream_result_to(result, target, request_id),
        result,
        label="stream event",
    )


def _stream_result_to(result: JsonDict, target: A2AVersion, request_id: RequestId) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_stream_response,
        to_core_stream_response,
        types_v03,
    )

    if target == "1.0":
        event: Final = _validate_stream_event(result)
        wrapper: Final = types_v03.SendStreamingMessageSuccessResponse(
            id=str(request_id) if request_id is not None else "",
            result=event,  # pyright: ignore[reportArgumentType]
        )
        return MessageToDict(to_core_stream_response(wrapper), preserving_proto_field_name=False)

    pb: Final = pb2_v10.StreamResponse()
    ParseDict(result, pb, ignore_unknown_fields=True)
    return _dump_03(to_compat_stream_response(pb, request_id).result)


def _convert_agent_card(card: JsonDict, target: A2AVersion) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_agent_card,
        to_core_agent_card,
        types_v03,
    )

    if target == "0.3":
        pb: Final = pb2_v10.AgentCard()
        ParseDict(card, pb, ignore_unknown_fields=True)
        lowered: Final = _dump_03(to_compat_agent_card(pb))
        lowered.pop("additionalInterfaces", None)
        return lowered

    core: Final = to_core_agent_card(types_v03.AgentCard.model_validate(card))
    return MessageToDict(core, preserving_proto_field_name=False)


def _validate_message_or_task(result: JsonDict) -> BaseModel:
    from a2a.compat.v0_3.conversions import types_v03

    if result.get("kind") == "task":
        return types_v03.Task.model_validate(result)
    return types_v03.Message.model_validate(result)


def _validate_stream_event(result: JsonDict) -> BaseModel:
    from a2a.compat.v0_3.conversions import types_v03

    kind: Final = result.get("kind")
    if kind == "task":
        return types_v03.Task.model_validate(result)
    if kind == "status-update":
        return types_v03.TaskStatusUpdateEvent.model_validate(result)
    if kind == "artifact-update":
        return types_v03.TaskArtifactUpdateEvent.model_validate(result)
    return types_v03.Message.model_validate(result)


def _lower_request_params(params: JsonDict, *, method: str) -> JsonDict:
    if method == "tasks/list":
        return _lower_list_tasks_params(params)

    from a2a.compat.v0_3.conversions import (
        ParseDict,
        pb2_v10,
        to_compat_cancel_task_request,
        to_compat_create_task_push_notification_config_request,
        to_compat_delete_task_push_notification_config_request,
        to_compat_get_task_push_notification_config_request,
        to_compat_get_task_request,
        to_compat_list_task_push_notification_config_request,
        to_compat_subscribe_to_task_request,
    )

    lowerings: Final[dict[str, Callable[[JsonDict], BaseModel]]] = {
        "tasks/get": lambda p: to_compat_get_task_request(_parse(ParseDict, p, pb2_v10.GetTaskRequest()), "").params,
        "tasks/cancel": lambda p: (
            to_compat_cancel_task_request(_parse(ParseDict, p, pb2_v10.CancelTaskRequest()), "").params
        ),
        "tasks/resubscribe": lambda p: (
            to_compat_subscribe_to_task_request(_parse(ParseDict, p, pb2_v10.SubscribeToTaskRequest()), "").params
        ),
        "tasks/pushNotificationConfig/set": lambda p: (
            to_compat_create_task_push_notification_config_request(
                _parse(
                    ParseDict,
                    _flatten_create_push_notification_params(p),
                    pb2_v10.TaskPushNotificationConfig(),
                ),
                "",
            ).params
        ),
        "tasks/pushNotificationConfig/get": lambda p: (
            to_compat_get_task_push_notification_config_request(
                _parse(ParseDict, p, pb2_v10.GetTaskPushNotificationConfigRequest()), ""
            ).params
        ),
        "tasks/pushNotificationConfig/list": lambda p: (
            to_compat_list_task_push_notification_config_request(
                _parse(ParseDict, p, pb2_v10.ListTaskPushNotificationConfigsRequest()), ""
            ).params
        ),
        "tasks/pushNotificationConfig/delete": lambda p: (
            to_compat_delete_task_push_notification_config_request(
                _parse(ParseDict, p, pb2_v10.DeleteTaskPushNotificationConfigRequest()), ""
            ).params
        ),
    }
    lower: Final = lowerings.get(method)
    if lower is None:
        return params
    return _dump_03(lower(params))


def _lower_list_tasks_params(params: JsonDict) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        types_v03,
    )

    proto: Final = pb2_v10.ListTasksRequest()
    _parse(ParseDict, params, proto)
    lowered: Final = MessageToDict(proto, preserving_proto_field_name=False)
    status_name: Final = str(pb2_v10.TaskState.Name(proto.status))
    valid_0_3_values: Final = frozenset(str(member.value) for member in types_v03.TaskState)
    compat_status: Final = _proto_task_state_name_to_0_3(status_name, valid_0_3_values)
    if compat_status is None:
        lowered.pop("status", None)
    else:
        lowered["status"] = compat_status
    return lowered


def _proto_task_state_name_to_0_3(name: str, valid_0_3_values: frozenset[str]) -> str | None:
    """Map a 1.0 protobuf ``TaskState`` enum name to its 0.3 wire string.

    The ``TASK_STATE_<NAME>`` enum names line up with the 0.3 wire values once the
    prefix is dropped and underscores become dashes, so no private SDK mapping is
    needed. The result is validated against the 0.3 enum's own values; an unspecified
    or unrecognized state yields ``None`` so the status filter is dropped.
    """
    base: Final = name.removeprefix("TASK_STATE_")
    if base == "UNSPECIFIED":
        return None
    candidate: Final = base.lower().replace("_", "-")
    return candidate if candidate in valid_0_3_values else None


def _flatten_create_push_notification_params(params: JsonDict) -> JsonDict:
    """Merge 1.x create envelope fields (parent/configId/config) into flat pb fields."""
    flat: Final = dict(params)
    config: Final = flat.pop("config", None)
    push_config: Final = flat.pop("pushNotificationConfig", None)
    nested: Final = config if config is not None else push_config
    if not isinstance(nested, dict):
        return params
    parent: Final = flat.pop("parent", None)
    if isinstance(parent, str) and parent.startswith("tasks/") and "taskId" not in flat:
        flat["taskId"] = parent.removeprefix("tasks/").split("/")[0]
    if (config_id := flat.pop("configId", None)) and "id" not in nested:
        nested["id"] = config_id
    flat.update(nested)
    return flat


def _parse(parse_dict: Callable[..., object], data: JsonDict, message: object) -> object:
    parse_dict(data, message, ignore_unknown_fields=True)
    return message
