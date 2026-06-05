"""
Load environment variables from .env file
"""
import os
from pathlib import Path


def load_env():
    """
    Load environment variables from .env file
    Simple implementation without python-dotenv dependency
    """
    env_path = Path(__file__).parent / '.env'
    
    if not env_path.exists():
        print(f"⚠️ .env file not found at {env_path}")
        print(f"💡 Copy .env.example to .env and fill in your values")
        return False
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Skip empty values
                if value:
                    os.environ[key] = value
    
    return True


def get_env(key: str, default: str = None) -> str:
    """
    Get environment variable with fallback to default
    """
    return os.getenv(key, default)


# Auto-load on import
if __name__ != '__main__':
    load_env()
