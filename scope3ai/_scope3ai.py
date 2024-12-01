import importlib.metadata
import importlib.util
from dataclasses import dataclass, field
import logging
from typing import Optional, Union
import dotenv
import os
from packaging.version import Version

from scope3ai.api import ImpactMetrics, ImpactRow, Scope3API
from scope3ai.exceptions import Scope3AIError
from scope3ai.log import logger

dotenv.load_dotenv()

SCOPE3API_BASE_URL = "https://aiapi.scope3.com"

def init_openai_instrumentor() -> None:
    if importlib.util.find_spec("openai") is not None:
        from scope3ai.tracers.openai_tracer import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument()


def init_anthropic_instrumentor() -> None:
    if importlib.util.find_spec("anthropic") is not None:
        from scope3ai.tracers.anthropic_tracer import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument()


def init_mistralai_instrumentor() -> None:
    if importlib.util.find_spec("mistralai") is not None:
        version = Version(importlib.metadata.version("mistralai"))
        if version < Version("1.0.0"):
            logger.warning(
                "MistralAI client v0.*.* will soon no longer be supported by Scope3AI."
            )
            from scope3ai.tracers.mistralai_tracer_v0 import MistralAIInstrumentor
        else:
            from scope3ai.tracers.mistralai_tracer_v1 import MistralAIInstrumentor

        instrumentor = MistralAIInstrumentor()
        instrumentor.instrument()


def init_huggingface_instrumentor() -> None:
    if importlib.util.find_spec("huggingface_hub") is not None:
        version = Version(importlib.metadata.version("huggingface_hub"))
        if version >= Version("0.22.0"):
            from scope3ai.tracers.huggingface_tracer import HuggingfaceInstrumentor

            instrumentor = HuggingfaceInstrumentor()
            instrumentor.instrument()


def init_cohere_instrumentor() -> None:
    if importlib.util.find_spec("cohere") is not None:
        from scope3ai.tracers.cohere_tracer import CohereInstrumentor

        instrumentor = CohereInstrumentor()
        instrumentor.instrument()


def init_google_instrumentor() -> None:
    if (
        importlib.util.find_spec("google") is not None
        and importlib.util.find_spec("google.generativeai") is not None
    ):
        from scope3ai.tracers.google_tracer import GoogleInstrumentor

        instrumentor = GoogleInstrumentor()
        instrumentor.instrument()


def init_litellm_instrumentor() -> None:
    if importlib.util.find_spec("litellm") is not None:
        from scope3ai.tracers.litellm_tracer import LiteLLMInstrumentor

        instrumentor = LiteLLMInstrumentor()
        instrumentor.instrument()


_INSTRUMENTS = {
    "openai": init_openai_instrumentor,
    "anthropic": init_anthropic_instrumentor,
    "mistralai": init_mistralai_instrumentor,
    "huggingface_hub": init_huggingface_instrumentor,
    "cohere": init_cohere_instrumentor,
    "google": init_google_instrumentor,
    "litellm": init_litellm_instrumentor,
}


class Scope3AI:
    """
    Scope3AI instrumentor to initialize function patching for each provider.

    Manages impact metrics tracking and provider initialization automatically.

    Examples:
        ```python
        from scope3ai import Scope3AI
        from openai import OpenAI

        scope3 = Scope3AI(api_key="your-api-key")

        client = OpenAI(api_key="<OPENAI_API_KEY>")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a funny joke!"}]
        )

        # Get total impact metrics
        total_impact = scope3.total_impact
        print(f"Total energy consumption: {total_impact.energy.value} kWh")
        print(f"Total GHG emissions: {total_impact.gwp.value} kgCO2eq")
        ```
    """

    @dataclass
    class _Config:
        api_key: str = field(default="")
        providers: list[str] = field(default_factory=list)
        api: Optional[Scope3API] = None
        logger: Optional[logging.Logger] = None
        live_results: bool = False
        environment: Optional[str] = None
        _total_impact: ImpactMetrics = field(default_factory=ImpactMetrics)

    _instance = None
    _config = _Config()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        api_key: str,
        live_results: bool = False,
        providers: Optional[Union[str, list[str]]] = None,
        base_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        environment: Optional[str] = None,
    ):
        """
        Initialize Scope3AI with automatic provider setup and metrics tracking.

        Args:
            api_key: API key for Scope3AI service
            live_results: Whether to fetch results live
            providers: List of providers to initialize (all supported providers by default)
            base_url: Optional base URL for the API
        """
        # Skip initialization if already initialized with same parameters
        if (
            self._config.api_key == api_key
            and self._config.live_results == live_results
            and base_url is None
        ):  # Only check base_url if it's being set
            return

        if isinstance(providers, str):
            providers = [providers]
        if providers is None:
            providers = list(_INSTRUMENTS.keys())

        # Initialize impact metrics if not already initialized
        if self._config._total_impact is None:
            self._config._total_impact = ImpactMetrics()

        self._init_providers(providers)

        self._config.api_key = api_key
        self._config.providers += providers
        self._config.providers = list(set(self._config.providers))
        self._config.live_results = live_results
        self._config.environment = environment
        if base_url is None:
            if os.getenv("SCOPE3AI_BASE_URL") is not None:
                base_url = os.getenv("SCOPE3AI_BASE_URL")
            else:
                base_url = SCOPE3API_BASE_URL
        if logger is not None:
            self._config.logger = logger
        else:
            self._config.logger = logging.getLogger("scope3ai")
        print(base_url)
        self._config.api = Scope3API(api_key, base_url, self._config.logger, environment=environment)


    def _init_providers(self, providers: list[str]) -> None:
        """Initialize the specified providers."""
        for provider in providers:
            if provider not in _INSTRUMENTS:
                raise Scope3AIError(
                    f"Could not find tracer for the `{provider}` provider."
                )
            if provider not in self._config.providers:
                init_func = _INSTRUMENTS[provider]
                init_func()

    def add_row(self, row: ImpactRow) -> Optional[ImpactMetrics]:
        """
        Add a new impact row to the total impact metrics.

        Args:
            row: ImpactRow object to add to the total

        Returns:
            Total impact metrics after adding the row
        """
        if self._config.live_results:
            impact = self._config.api.record_inferences([row])
            self.add_impact(impact)
            return impact
        else:
            # todo - make an async (or at least non-waiting) version of this
            self._config.api.record_inferences([row])
            return None

    def add_impact(self, impact: ImpactMetrics) -> None:
        """
        Add new impact metrics to the total.

        Args:
            impact: ImpactMetrics object to add to the total
        """
        if self._config._total_impact is None:
            self._config._total_impact = impact
        else:
            self._config._total_impact += impact

    @property
    def total_impact(self) -> ImpactMetrics:
        """Get the current total impact metrics if live results are enabled."""
        if not self._config.live_results:
            raise Scope3AIError(
                "Live results are not enabled. Please set `live_results=True` to fetch the total impact."
            )
        if self._config._total_impact is None:
            self._config._total_impact = ImpactMetrics()
        return self._config._total_impact

    @property
    def live_results(self) -> bool:
        """Get the current live results setting."""
        return self._config.live_results

    @property
    def api(self) -> Optional[Scope3API]:
        """Get the current API instance."""
        return self._config.api


def init_instruments(providers: list[str]) -> None:
    for provider in providers:
        if provider not in _INSTRUMENTS:
            raise Scope3AIError(f"Could not find tracer for the `{provider}` provider.")
        if provider not in Scope3AI.config.providers:
            init_func = _INSTRUMENTS[provider]
            init_func()
