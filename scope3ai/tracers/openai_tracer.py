import time
from typing import Any, Callable, Optional, Union

from wrapt import wrap_function_wrapper

from scope3ai._scope3ai import Scope3AI
from scope3ai.impacts import Impacts
from scope3ai.tracers.utils import llm_impacts
from scope3ai.api import ImpactRow, Model, ImpactMetrics

import json

try:
    from openai import AsyncStream, Stream
    from openai.resources.chat import AsyncCompletions, Completions
    from openai.types.chat import ChatCompletion as _ChatCompletion
    from openai.types.chat import ChatCompletionChunk as _ChatCompletionChunk
except ImportError:
    AsyncStream = object()
    Stream = object()
    AsyncCompletions = object()
    Completions = object()
    _ChatCompletion = object()
    _ChatCompletionChunk = object()


PROVIDER = "openai"


class ChatCompletion(_ChatCompletion):
    impact: Optional[ImpactMetrics] = None


class ChatCompletionChunk(_ChatCompletionChunk):
    impact: Optional[ImpactMetrics] = None


def openai_chat_wrapper(
    wrapped: Callable,
    instance: Completions,
    args: Any,
    kwargs: Any
) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
    if kwargs.get("stream", False):
        return openai_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return openai_chat_wrapper_non_stream(wrapped, instance, args, kwargs)


def openai_chat_wrapper_non_stream(
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start

    Scope3AI._config.logger.debug(f"Response from OpenAI: {response}")

    model_requested = kwargs["model"]
    model_used = response.model

    scope3_row = ImpactRow(
        model=Model(
            id=model_requested
        ),
        model_used=Model(
            id=model_used
        ),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        request_duration_ms=request_latency * 1000, # TODO: can we get the header that has the processing time
        managed_service_id=PROVIDER,
    )
    
    Scope3AI._instance.add_row(scope3_row)
    return response


def openai_chat_wrapper_stream(
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> Stream[ChatCompletionChunk]:
    timer_start = time.perf_counter()
    if "stream_options" not in kwargs:
        kwargs["stream_options"] = {}
    if "include_usage" not in kwargs["stream_options"]:
        kwargs["stream_options"]["include_usage"] = True
    elif kwargs["stream_options"] == False:
        raise ValueError("stream_options include_usage must be True")
    stream = wrapped(*args, **kwargs)
    for i, chunk in enumerate(stream):
        request_latency = time.perf_counter() - timer_start
        if chunk.usage is not None:
            Scope3AI._config.logger.debug(f"Last chunk from OpenAI: {chunk}")

            model_requested = kwargs["model"]
            model_used = chunk.model

            scope3_row = ImpactRow(
                model=Model(
                    id=model_requested
                ),
                model_used=Model(
                    id=model_used
                ),
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                request_duration_ms=request_latency * 1000, # TODO: can we get the header that has the processing time
                managed_service_id=PROVIDER,
            )
            
            Scope3AI._instance.add_row(scope3_row)
        
        yield chunk


async def openai_async_chat_wrapper(
    wrapped: Callable,
    instance: AsyncCompletions,
    args: Any,
    kwargs: Any,
) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
    if kwargs.get("stream", False):
        return openai_async_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return await openai_async_chat_wrapper_base(wrapped, instance, args, kwargs)


async def openai_async_chat_wrapper_base(
    wrapped: Callable,
    instance: AsyncCompletions,     # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = response.model
    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=response.usage.completion_tokens,
        request_latency=request_latency,
        electricity_mix_zone=Scope3AI.config.electricity_mix_zone
    )
    if impacts is not None:
        return ChatCompletion(**response.model_dump(), impacts=impacts)
    else:
        return response


async def openai_async_chat_wrapper_stream(
    wrapped: Callable,
    instance: AsyncCompletions,     # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> AsyncStream[ChatCompletionChunk]:
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    i = 0
    token_count = 0
    async for chunk in stream:
        if i == 0 and chunk.model == "":
            continue
        if i > 0 and chunk.choices[0].finish_reason is None:
            token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_name = chunk.model
        impacts = llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=token_count,
            request_latency=request_latency,
            electricity_mix_zone=Scope3AI.config.electricity_mix_zone
        )
        if impacts is not None:
            yield ChatCompletionChunk(**chunk.model_dump(), impacts=impacts)
        else:
            yield chunk
        i += 1


class OpenAIInstrumentor:
    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": "openai.resources.chat.completions",
                "name": "Completions.create",
                "wrapper": openai_chat_wrapper,
            },
            {
                "module": "openai.resources.chat.completions",
                "name": "AsyncCompletions.create",
                "wrapper": openai_async_chat_wrapper,
            },
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
