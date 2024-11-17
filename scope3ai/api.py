
# On success events, log the event to the Scope3 API to measure the carbon footprint of the API call.

import logging
from datetime import datetime, timezone

import requests

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class Task(str, Enum):
    TEXT_GENERATION = "text-generation"
    CHAT = "chat"
    TEXT_EMBEDDING = "text-embedding"
    # ... add other tasks as needed

class DataType(str, Enum):
    FP8 = "fp8"
    FP16 = "fp16"
    FP32 = "fp32"
    # ... add other data types as needed

class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ORACLE = "oracle"
    IBM = "ibm"

class ManagedServiceProvider(str, Enum):
    AWS_BEDROCK = "aws-bedrock"
    AZURE_ML = "azure-ml"
    GOOGLE_VERTEX = "google-vertex"
    IBM_WATSON = "ibm-watson"
    HUGGING_FACE = "hugging-face"

class GPU(BaseModel):
    """Configuration for GPU hardware"""
    name: Optional[str] = Field(None, description="GPU name (e.g., 'NVIDIA A100 40GB')")
    id: Optional[str] = Field(None, description="GPU identifier (e.g., 'a100_40gb')")
    max_power_w: Optional[float] = Field(None, description="Maximum power consumption in watts")
    embodied_emissions_kgco2e: Optional[float] = None
    embodied_water_mlh2o: Optional[float] = None
    performance_ratio_to_h200: Optional[float] = None

class Node(BaseModel):
    """Configuration for compute node"""
    id: Optional[str] = None
    cloud_provider: Optional[CloudProvider] = Field(None, description="Cloud provider (e.g., 'aws')")
    cloud_instance_id: Optional[str] = Field(None, description="Cloud instance type (e.g., 'a2-highgpu-1g')")
    managed_service: Optional[ManagedServiceProvider] = None
    gpu: Optional[GPU] = None
    gpu_count: Optional[int] = Field(None, ge=0, le=10000)
    cpu_count: Optional[int] = Field(None, ge=1, le=10000)
    idle_power_w_ex_gpu: Optional[float] = Field(None, ge=0, le=10000)
    average_utilization_rate: Optional[float] = Field(None, ge=0, le=1)
    embodied_emissions_kgco2e_ex_gpu: Optional[float] = Field(None, ge=0, le=100000)
    embodied_water_l_ex_gpu: Optional[float] = Field(None, ge=0, le=100000)
    use_life_years: Optional[float] = Field(None, ge=1, le=30)

class Model(BaseModel):
    """Configuration for AI model"""
    id: str = Field(..., description="Model identifier")
    name: Optional[str] = None
    family: Optional[str] = None
    hugging_face_path: Optional[str] = None
    benchmark_model_id: Optional[str] = None
    total_params_billions: Optional[float] = None
    number_of_experts: Optional[int] = None
    params_per_expert_billions: Optional[float] = None
    tensor_parallelism: Optional[int] = None
    datatype: Optional[DataType] = None
    task: Optional[Task] = None
    training_usage_energy_kwh: Optional[float] = None
    training_usage_emissions_kgco2e: Optional[float] = None
    training_usage_water_l: Optional[float] = None
    training_embodied_emissions_kgco2e: Optional[float] = None
    training_embodied_water_l: Optional[float] = None
    estimated_use_life_days: Optional[float] = None
    estimated_requests_per_day: Optional[float] = None
    fine_tuned_from_model_id: Optional[str] = None

class ImageDimensions(BaseModel):
    """Image dimensions specification"""
    dimensions: str = Field(..., pattern=r"^(\d{1,4})x(\d{1,4})$")

class LocationConfig(BaseModel):
    """Geographic location configuration"""
    cloud_region: Optional[str] = Field(None, description="Cloud region (e.g., 'us-east-1')")
    country: Optional[str] = Field(None, pattern="^[A-Z]{2}$", min_length=2, max_length=2,
                                 description="Two-letter country code (e.g., 'US')")
    region: Optional[str] = Field(None, pattern="^[A-Z]{2}$", min_length=2, max_length=2,
                                description="Two-letter region code (e.g., 'VA')")    

class ImpactRow(BaseModel):
    """Complete input for an impact request"""
    model: Model
    location: Optional[LocationConfig] = None
    node: Optional[Node] = None
    utc_timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp for the request"
    )
    """Context of the request"""
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    client_id: Optional[str] = None
    project_id: Optional[str] = None
    application_id: Optional[str] = None
    """Metrics about the model usage"""
    task: Optional[Task] = None
    input_tokens: Optional[int] = Field(None, ge=0, le=100000000)
    output_tokens: Optional[int] = Field(None, ge=0, le=100000000)
    input_audio_seconds: Optional[int] = Field(None, ge=0, le=100000)
    input_images: Optional[List[ImageDimensions]] = Field(None, max_items=100)
    input_steps: Optional[int] = Field(None, ge=1, le=10000)
    output_images: Optional[List[ImageDimensions]] = Field(None, max_items=100)
    output_video_frames: Optional[int] = Field(None, ge=0, le=100000000)
    output_video_resolution: Optional[int] = None

class ImpactMetrics(BaseModel):
    """Impact metrics for the model usage"""
    usage_energy_wh: Optional[float] = Field(0, ge=0, le=100000000)
    usage_emissions_gco2e: Optional[float] = Field(0, ge=0, le=100000000)
    usage_water_ml: Optional[float] = Field(0, ge=0, le=100000000)
    embodied_emissions_gco2e: Optional[float] = Field(0, ge=0, le=100000000)
    embodied_water_ml: Optional[float] = Field(0, ge=0, le=100000000)

    def __add__(self, other: 'ImpactMetrics') -> 'ImpactMetrics':
        if not isinstance(other, ImpactMetrics):
            raise ValueError("Can only add ImpactMetrics with another ImpactMetrics instance")
        
        return ImpactMetrics(
            usage_energy_wh=(self.usage_energy_wh or 0) + (other.usage_energy_wh or 0),
            usage_emissions_gco2e=(self.usage_emissions_gco2e or 0) + (other.usage_emissions_gco2e or 0),
            usage_water_ml=(self.usage_water_ml or 0) + (other.usage_water_ml or 0),
            embodied_emissions_gco2e=(self.embodied_emissions_gco2e or 0) + (other.embodied_emissions_gco2e or 0),
            embodied_water_ml=(self.embodied_water_ml or 0) + (other.embodied_water_ml or 0)
        )
    
class ImpactResponseRow(BaseModel):
    """Single row of impact data from the API response"""
    fine_tuning_impact: ImpactMetrics
    inference_impact: ImpactMetrics
    training_impact: ImpactMetrics
    total_impact: ImpactMetrics

class ImpactResponse(BaseModel):
    """Complete response from the impact API"""
    has_errors: bool
    rows: List[ImpactResponseRow]

class ImpactRequest(BaseModel):
    """Final request structure for the API"""
    rows: List[ImpactRow] = Field(..., max_items=1000)


class Scope3API():

    def __init__(self, access_token: str, api_base: str, logger: logging.Logger, environment: str = None, integration_source: str = "scope3ai-python-openai"):
        self.access_token = access_token
        self.api_base = api_base
        self.logger = logger
        self.integration_source = integration_source
        self.environment = environment

        
    def record_inferences(
        self,
        inferences: List[ImpactRow],
        debug: bool = False,
    ) -> ImpactMetrics:
        """
        Record inferences using the Scope3 API, and possibly return results live

        Args:
            inferences: List of inferences to record
            debug: Whether to include debug information in the response

        Returns:
            ImpactMetrics: The impact response from the API
        """
        # Convert the input to the API's expected format
        impact_request = ImpactRequest(rows=inferences, environment=self.environment, integration_source=self.integration_source)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        
        url = f"{self.api_base}/v1/impact"
        if debug:
            url = f"{url}?debug=true"

        body = impact_request.model_dump_json(exclude_none=True, exclude_unset=True)
        
        try:
            response = requests.post(
                url=url,
                data=body,
                headers=headers,
            )
            response.raise_for_status()
            # self.logger.debug(f"Response: {response.text}")

            impact_response = ImpactResponse.model_validate(response.json())
            sum = ImpactMetrics()
            for impact_row in impact_response.rows:
                sum += impact_row.total_impact
            return sum
            
        except Exception as e:
            error_response = getattr(e, "response", None)
            if error_response is not None and hasattr(error_response, "text"):
                self.logger.error(f"\nError Message: {error_response.text}")
            raise e
        