from confz import BaseConfig, CLArgSource

# Step 1: Define the configuration class
class MyConfig(BaseConfig):
    # You can define default values and types here
    host: str
    port: int = 8080


def load_config():
    # Step 3: Load the configuration
    config = MyConfig(
        config_sources=[CLArgSource()]
    )
    print(f"Server starting at {config.host}:{config.port}")
    return config

# Step 3: Load the configuration
if __name__ == "__main__":
    config = load_config()
    print(f"Server starting at {config.host}:{config.port}")
