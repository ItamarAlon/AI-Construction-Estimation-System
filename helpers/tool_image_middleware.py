"""Relocate images out of tool messages into a following user message.

OpenAI's chat API rejects image blocks inside a 'tool'-role message ("Image URLs
are only allowed for messages with role 'user'"). Tools like list_colored_segments
return per-segment image crops, so before each model call we rewrite the request:
the tool message keeps a short text pointer, and its full content (text lines +
image crops, in order) is moved into a HumanMessage inserted right after the tool
results. This only affects what is sent to the model, not the persisted state.
"""
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import HumanMessage, ToolMessage


def _has_image(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in content
    )


@wrap_model_call
def relocate_tool_images(request, handler):
    messages = request.messages
    if not any(isinstance(m, ToolMessage) and _has_image(m.content) for m in messages):
        return handler(request)  # nothing to do — avoid rebuilding the list

    new_messages = []
    pending: list[HumanMessage] = []
    for m in messages:
        is_tool = isinstance(m, ToolMessage)
        # Flush relocated images once the contiguous run of tool results ends, so
        # we never split an assistant's tool_call from its tool responses.
        if not is_tool and pending:
            new_messages.extend(pending)
            pending = []

        if is_tool and _has_image(m.content):
            new_messages.append(
                ToolMessage(
                    content="(segment listing with per-segment image crops is in the next message)",
                    tool_call_id=m.tool_call_id,
                    name=m.name,
                    status=m.status,
                )
            )
            pending.append(HumanMessage(content=m.content))
        else:
            new_messages.append(m)

    if pending:
        new_messages.extend(pending)

    return handler(request.override(messages=new_messages))
