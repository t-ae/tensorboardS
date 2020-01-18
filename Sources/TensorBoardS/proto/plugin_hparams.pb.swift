// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: proto/plugin_hparams.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================

// Defines protos for storing a hypertuning experiment data inside Summary tags.
//
// A hypertuning-experiment data consists of metadata that's constant
// throughout the experiment and evolving metric data for each training session
// in the experiment. The HParams plugin assumes the following organization of
// this entire data set. Experiment metadata is recorded in the empty run in a
// tag (named by the Python constant) metadata.EXPERIMENT_TAG. Within the
// experiment, for a session named by <session_name> its metadata is recorded
// in the run <session_name> in the tags metadata.SESSION_START_INFO and
// metadata.SESSION_END_INFO. Finally, the session's metric data for a metric
// with a (<group>, <tag>) name (see MetricName in api.proto), is recorded
// in a Scalar-plugin summary with tag <tag> in the run <session_name><group>.

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that your are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

/// HParam summaries created by `tensorboard.plugins.hparams.summary`
/// module will include `SummaryMetadata` whose `plugin_data` field has
/// as `content` a serialized HParamsPluginData message.
struct TensorBoardS_Hparam_HParamsPluginData {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// The version of the plugin data schema.
  var version: Int32 {
    get {return _storage._version}
    set {_uniqueStorage()._version = newValue}
  }

  var data: OneOf_Data? {
    get {return _storage._data}
    set {_uniqueStorage()._data = newValue}
  }

  var experiment: TensorBoardS_Hparam_Experiment {
    get {
      if case .experiment(let v)? = _storage._data {return v}
      return TensorBoardS_Hparam_Experiment()
    }
    set {_uniqueStorage()._data = .experiment(newValue)}
  }

  var sessionStartInfo: TensorBoardS_Hparam_SessionStartInfo {
    get {
      if case .sessionStartInfo(let v)? = _storage._data {return v}
      return TensorBoardS_Hparam_SessionStartInfo()
    }
    set {_uniqueStorage()._data = .sessionStartInfo(newValue)}
  }

  var sessionEndInfo: TensorBoardS_Hparam_SessionEndInfo {
    get {
      if case .sessionEndInfo(let v)? = _storage._data {return v}
      return TensorBoardS_Hparam_SessionEndInfo()
    }
    set {_uniqueStorage()._data = .sessionEndInfo(newValue)}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  enum OneOf_Data: Equatable {
    case experiment(TensorBoardS_Hparam_Experiment)
    case sessionStartInfo(TensorBoardS_Hparam_SessionStartInfo)
    case sessionEndInfo(TensorBoardS_Hparam_SessionEndInfo)

  #if !swift(>=4.1)
    static func ==(lhs: TensorBoardS_Hparam_HParamsPluginData.OneOf_Data, rhs: TensorBoardS_Hparam_HParamsPluginData.OneOf_Data) -> Bool {
      switch (lhs, rhs) {
      case (.experiment(let l), .experiment(let r)): return l == r
      case (.sessionStartInfo(let l), .sessionStartInfo(let r)): return l == r
      case (.sessionEndInfo(let l), .sessionEndInfo(let r)): return l == r
      default: return false
      }
    }
  #endif
  }

  init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

struct TensorBoardS_Hparam_SessionStartInfo {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// A map describing the hyperparameter values for the session.
  /// Maps each hyperparameter name to its value.
  /// Currently only scalars are supported.
  var hparams: Dictionary<String,SwiftProtobuf.Google_Protobuf_Value> = [:]

  /// A URI for where checkpoints are saved.
  var modelUri: String = String()

  /// An optional URL to a website monitoring the session.
  var monitorURL: String = String()

  /// The name of the session group containing this session. If empty, the
  /// group name is taken to be the session id (so this session is the only
  /// member of its group).
  var groupName: String = String()

  /// The time the session started in seconds since epoch.
  var startTimeSecs: Double = 0

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

struct TensorBoardS_Hparam_SessionEndInfo {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var status: TensorBoardS_Hparam_Status = .unknown

  /// The time the session ended in seconds since epoch.
  var endTimeSecs: Double = 0

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "TensorBoardS.hparam"

extension TensorBoardS_Hparam_HParamsPluginData: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".HParamsPluginData"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "version"),
    2: .same(proto: "experiment"),
    3: .standard(proto: "session_start_info"),
    4: .standard(proto: "session_end_info"),
  ]

  fileprivate class _StorageClass {
    var _version: Int32 = 0
    var _data: TensorBoardS_Hparam_HParamsPluginData.OneOf_Data?

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _version = source._version
      _data = source._data
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    _ = _uniqueStorage()
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularInt32Field(value: &_storage._version)
        case 2:
          var v: TensorBoardS_Hparam_Experiment?
          if let current = _storage._data {
            try decoder.handleConflictingOneOf()
            if case .experiment(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._data = .experiment(v)}
        case 3:
          var v: TensorBoardS_Hparam_SessionStartInfo?
          if let current = _storage._data {
            try decoder.handleConflictingOneOf()
            if case .sessionStartInfo(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._data = .sessionStartInfo(v)}
        case 4:
          var v: TensorBoardS_Hparam_SessionEndInfo?
          if let current = _storage._data {
            try decoder.handleConflictingOneOf()
            if case .sessionEndInfo(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._data = .sessionEndInfo(v)}
        default: break
        }
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if _storage._version != 0 {
        try visitor.visitSingularInt32Field(value: _storage._version, fieldNumber: 1)
      }
      switch _storage._data {
      case .experiment(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      case .sessionStartInfo(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
      case .sessionEndInfo(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
      case nil: break
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: TensorBoardS_Hparam_HParamsPluginData, rhs: TensorBoardS_Hparam_HParamsPluginData) -> Bool {
    if lhs._storage !== rhs._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((lhs._storage, rhs._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let rhs_storage = _args.1
        if _storage._version != rhs_storage._version {return false}
        if _storage._data != rhs_storage._data {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension TensorBoardS_Hparam_SessionStartInfo: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".SessionStartInfo"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "hparams"),
    2: .standard(proto: "model_uri"),
    3: .standard(proto: "monitor_url"),
    4: .standard(proto: "group_name"),
    5: .standard(proto: "start_time_secs"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMessageMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.Google_Protobuf_Value>.self, value: &self.hparams)
      case 2: try decoder.decodeSingularStringField(value: &self.modelUri)
      case 3: try decoder.decodeSingularStringField(value: &self.monitorURL)
      case 4: try decoder.decodeSingularStringField(value: &self.groupName)
      case 5: try decoder.decodeSingularDoubleField(value: &self.startTimeSecs)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.hparams.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMessageMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.Google_Protobuf_Value>.self, value: self.hparams, fieldNumber: 1)
    }
    if !self.modelUri.isEmpty {
      try visitor.visitSingularStringField(value: self.modelUri, fieldNumber: 2)
    }
    if !self.monitorURL.isEmpty {
      try visitor.visitSingularStringField(value: self.monitorURL, fieldNumber: 3)
    }
    if !self.groupName.isEmpty {
      try visitor.visitSingularStringField(value: self.groupName, fieldNumber: 4)
    }
    if self.startTimeSecs != 0 {
      try visitor.visitSingularDoubleField(value: self.startTimeSecs, fieldNumber: 5)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: TensorBoardS_Hparam_SessionStartInfo, rhs: TensorBoardS_Hparam_SessionStartInfo) -> Bool {
    if lhs.hparams != rhs.hparams {return false}
    if lhs.modelUri != rhs.modelUri {return false}
    if lhs.monitorURL != rhs.monitorURL {return false}
    if lhs.groupName != rhs.groupName {return false}
    if lhs.startTimeSecs != rhs.startTimeSecs {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension TensorBoardS_Hparam_SessionEndInfo: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".SessionEndInfo"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "status"),
    2: .standard(proto: "end_time_secs"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularEnumField(value: &self.status)
      case 2: try decoder.decodeSingularDoubleField(value: &self.endTimeSecs)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.status != .unknown {
      try visitor.visitSingularEnumField(value: self.status, fieldNumber: 1)
    }
    if self.endTimeSecs != 0 {
      try visitor.visitSingularDoubleField(value: self.endTimeSecs, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: TensorBoardS_Hparam_SessionEndInfo, rhs: TensorBoardS_Hparam_SessionEndInfo) -> Bool {
    if lhs.status != rhs.status {return false}
    if lhs.endTimeSecs != rhs.endTimeSecs {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
