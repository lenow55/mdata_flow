from pathlib import Path
from typing import Annotated, ClassVar
from pydantic import BeforeValidator, Field, SecretStr
from pydantic.functional_validators import field_validator
from typing_extensions import override
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from mdata_flow.file_name_validator import FileNameValidator


def convert_str2int(v: str | int):
    if isinstance(v, str):
        v = int(v)
    return v


IntMapStr = Annotated[int, BeforeValidator(convert_str2int)]


class DataExperiment(BaseSettings):
    name: str
    artifact_path: str


class DatasetStoreSettings(BaseSettings):
    s3_endpoint_url: str = Field(default="http://localhost:9000")
    access_key_id: str
    secret_access_key: SecretStr
    data_experiment: DataExperiment
    tracking_uri: str
    local_cache: str

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        json_file=(
            "config.json",
            "debug_config.json",
        ),
    )

    @field_validator("local_cache", mode="after")
    @classmethod
    def set_cache_abspath(cls, path: str):
        if FileNameValidator.validate_with_pathlib(path):
            return Path(path).as_posix()
        else:
            raise ValueError("Bad cache filepath")

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
