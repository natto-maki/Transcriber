# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transcriber_service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19transcriber_service.proto\"n\n\x11TranscribeRequest\x12\x12\n\naudio_data\x18\x01 \x01(\x0c\x12\x10\n\x08language\x18\x02 \x01(\t\x12\x15\n\rget_embedding\x18\x03 \x01(\t\x12\x1c\n\x14min_segment_duration\x18\x04 \x01(\x02\":\n\x12TranscribeResponse\x12\x10\n\x08segments\x18\x01 \x01(\t\x12\x12\n\nembeddings\x18\x02 \x03(\x0c\"+\n\x15\x44\x65tectLanguageRequest\x12\x12\n\naudio_data\x18\x01 \x01(\x0c\"4\n\x16\x44\x65tectLanguageResponse\x12\x1a\n\x12\x64\x65tected_languages\x18\x01 \x01(\t2\x92\x01\n\x12TranscriberService\x12\x37\n\nTranscribe\x12\x12.TranscribeRequest\x1a\x13.TranscribeResponse\"\x00\x12\x43\n\x0e\x44\x65tectLanguage\x12\x16.DetectLanguageRequest\x1a\x17.DetectLanguageResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'transcriber_service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_TRANSCRIBEREQUEST']._serialized_start=29
  _globals['_TRANSCRIBEREQUEST']._serialized_end=139
  _globals['_TRANSCRIBERESPONSE']._serialized_start=141
  _globals['_TRANSCRIBERESPONSE']._serialized_end=199
  _globals['_DETECTLANGUAGEREQUEST']._serialized_start=201
  _globals['_DETECTLANGUAGEREQUEST']._serialized_end=244
  _globals['_DETECTLANGUAGERESPONSE']._serialized_start=246
  _globals['_DETECTLANGUAGERESPONSE']._serialized_end=298
  _globals['_TRANSCRIBERSERVICE']._serialized_start=301
  _globals['_TRANSCRIBERSERVICE']._serialized_end=447
# @@protoc_insertion_point(module_scope)
