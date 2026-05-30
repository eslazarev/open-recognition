"""Domain exceptions mapped to AWS-Rekognition error codes.

Each exception carries an `aws_code` and `http_status` so the wire
layer can serialize it into the JSON-1.1 error envelope without any
inline branching.
"""

from __future__ import annotations


class DomainError(Exception):
    aws_code: str = "InternalServerError"
    http_status: int = 500


class InvalidParameterValueError(DomainError):
    aws_code = "InvalidParameterException"
    http_status = 400


class InvalidImageFormatError(DomainError):
    aws_code = "InvalidImageFormatException"
    http_status = 400


class ImageTooLargeError(DomainError):
    aws_code = "ImageTooLargeException"
    http_status = 400


class InvalidS3ObjectError(DomainError):
    aws_code = "InvalidS3ObjectException"
    http_status = 400


class ResourceNotFoundError(DomainError):
    aws_code = "ResourceNotFoundException"
    http_status = 400


class ResourceAlreadyExistsError(DomainError):
    aws_code = "ResourceAlreadyExistsException"
    http_status = 400


class UnknownOperationError(DomainError):
    aws_code = "UnknownOperationException"
    http_status = 404
