# Track the environmental impact of your use of AI

## Installation

```shell
pip install scope3ai
```

## Usage

```python
from scope3ai import Scope3AI
from openai import OpenAI

# Initialize Scope3 with parameters
Scope3.init(log=false, environment=production, project="customer service")

client = OpenAI(api_key="<OPENAI_API_KEY>")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "How can I get more out of my subscription?"}
    ]
)

# Get environmental impact metrics
print(f"Estimated CO2e impact: {response.scope3.total_gco2e} g")
print(f"Estimated water impact: {response.scope3.total_mlh2o} ml")
print(f"Estimated energy use: {response.scope3.total_energy_wh} wh")

```

## Parameters

todo

## License

todo
