import base64
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.types.interactions import CancelInteractionResult, DeleteInteractionResult, InteractionsAPIResponse

_PREFIX: Final = "intr_"
_PAYLOAD: Final = re.compile(r"^litellm:custom_llm_provider:([^;]*);model_id:([^;]*);interaction_id:(.+)$")
_AFFINITY_RESPONSES: Final = (InteractionsAPIResponse, DeleteInteractionResult, CancelInteractionResult)
_STRING_OBJECT_MAP: Final = TypeAdapter(Mapping[str, object])


def _mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _STRING_OBJECT_MAP.validate_python(value)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class DecodedInteractionId:
    custom_llm_provider: str | None
    model_id: str | None
    interaction_id: str


def encode_interaction_id(*, custom_llm_provider: str | None, model_id: str, interaction_id: str) -> str:
    if decode_interaction_id(interaction_id).model_id is not None:
        return interaction_id
    provider: Final = custom_llm_provider or ""
    payload: Final = f"litellm:custom_llm_provider:{provider};model_id:{model_id};interaction_id:{interaction_id}"
    token: Final = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"{_PREFIX}{token}"


def decode_interaction_id(interaction_id: str) -> DecodedInteractionId:
    if not interaction_id.startswith(_PREFIX):
        return DecodedInteractionId(custom_llm_provider=None, model_id=None, interaction_id=interaction_id)
    padded: Final = interaction_id.removeprefix(_PREFIX)
    try:
        payload: Final = base64.urlsafe_b64decode(padded + "=" * (-len(padded) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        return DecodedInteractionId(custom_llm_provider=None, model_id=None, interaction_id=interaction_id)
    found: Final = _PAYLOAD.fullmatch(payload)
    if found is None:
        return DecodedInteractionId(custom_llm_provider=None, model_id=None, interaction_id=interaction_id)
    provider: Final = found.group(1) or None
    model_id: Final = found.group(2) or None
    return DecodedInteractionId(custom_llm_provider=provider, model_id=model_id, interaction_id=found.group(3))


def raw_interaction_id(interaction_id: object) -> str | None:
    if not isinstance(interaction_id, str):
        return None
    decoded: Final = decode_interaction_id(interaction_id)
    return decoded.interaction_id if decoded.model_id is not None else None


def deployment_id_of_interaction(interaction_id: object) -> str | None:
    if not isinstance(interaction_id, str):
        return None
    return decode_interaction_id(interaction_id).model_id


def creating_deployment_id(kwargs: Mapping[str, object]) -> str | None:
    model_info: Final = _mapping(kwargs.get("model_info"))
    if model_info is None:
        return None
    deployment_id: Final = model_info.get("id")
    return deployment_id if isinstance(deployment_id, str) and deployment_id else None


def _prefixed_deployment(bucket: object) -> str | None:
    parsed: Final = _mapping(bucket)
    if parsed is None:
        return None
    deployment: Final = parsed.get("deployment")
    return deployment if isinstance(deployment, str) and "/" in deployment else None


def creating_provider(kwargs: Mapping[str, object]) -> str | None:
    deployment: Final = _prefixed_deployment(kwargs.get("litellm_metadata")) or _prefixed_deployment(
        kwargs.get("metadata")
    )
    if deployment is None:
        return None
    provider, _, _ = deployment.partition("/")
    return provider or None


def with_client_interaction_id(response: object, interaction_id: str) -> object:
    if isinstance(response, _AFFINITY_RESPONSES):
        copied: Final = response.model_copy(update={"id": interaction_id})
        copied.hidden_params = dict(response.hidden_params)
        return copied
    body: Final = _mapping(response)
    if body is None or "id" not in body:
        return response
    return {**body, "id": interaction_id}


def client_facing_interaction(
    response: InteractionsAPIResponse, kwargs: Mapping[str, object]
) -> InteractionsAPIResponse:
    interaction_id: Final = response.id
    if not interaction_id:
        return response
    deployment_id: Final = creating_deployment_id(kwargs)
    if deployment_id is None:
        return response
    encoded: Final = encode_interaction_id(
        custom_llm_provider=creating_provider(kwargs),
        model_id=deployment_id,
        interaction_id=interaction_id,
    )
    if encoded == interaction_id:
        return response
    updated: Final = with_client_interaction_id(response, encoded)
    if isinstance(updated, InteractionsAPIResponse):
        return updated
    return response
