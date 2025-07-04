from typing import Optional, TypeVar, Generic

from pydantic import BaseModel

T = TypeVar("T")


class ResponseSchemaBase(BaseModel):
    __abstract__ = True

    code: str = ''
    message: str = ''

    def custom_response(self, code: str, message: str):
        self.code = code
        self.message = message
        return self

    def success_response(self):
        self.code = '000'
        self.message = 'Success'
        return self

class DataResponse(ResponseSchemaBase, BaseModel, Generic[T]):
    data: Optional[T] = None

    class Config:
        arbitrary_types_allowed = True
        from_attributes = True

    @classmethod
    def success_response(cls, data: T):
        return cls(
            code='000',
            message='Success',
            data=data
        )

    @classmethod
    def custom_response(cls, code: str, message: str, data: T):
        return cls(
            code=code,
            message=message,
            data=data
        )



class MetadataSchema(BaseModel):
    current_page: int
    page_size: int
    total_items: int

    class Config:
        from_attributes = True
