from pydantic import BaseModel, Field

class Arguments(BaseModel): 
    config: str = Field(description="config file")
    snapshot: str = Field(description="model name")
    video_name: str = Field(description="videos or image files")