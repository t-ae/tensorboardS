// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: proto/plugin_mesh.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

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

/// A MeshPluginData encapsulates information on which plugins are able to make
/// use of a certain summary value.
struct TensorBoardS_Mesh_MeshPluginData {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Version `0` is the only supported version.
  var version: Int32 = 0

  /// The name of the mesh summary this particular summary belongs to.
  var name: String = String()

  /// Type of data in the summary.
  var contentType: TensorBoardS_Mesh_MeshPluginData.ContentType = .undefined

  /// JSON-serialized dictionary of ThreeJS classes configuration.
  var jsonConfig: String = String()

  /// Shape of underlying data. Cache it here for performance reasons.
  var shape: [Int32] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  enum ContentType: SwiftProtobuf.Enum {
    typealias RawValue = Int
    case undefined // = 0
    case vertex // = 1

    /// Triangle face.
    case face // = 2
    case color // = 3
    case UNRECOGNIZED(Int)

    init() {
      self = .undefined
    }

    init?(rawValue: Int) {
      switch rawValue {
      case 0: self = .undefined
      case 1: self = .vertex
      case 2: self = .face
      case 3: self = .color
      default: self = .UNRECOGNIZED(rawValue)
      }
    }

    var rawValue: Int {
      switch self {
      case .undefined: return 0
      case .vertex: return 1
      case .face: return 2
      case .color: return 3
      case .UNRECOGNIZED(let i): return i
      }
    }

  }

  init() {}
}

#if swift(>=4.2)

extension TensorBoardS_Mesh_MeshPluginData.ContentType: CaseIterable {
  // The compiler won't synthesize support with the UNRECOGNIZED case.
  static var allCases: [TensorBoardS_Mesh_MeshPluginData.ContentType] = [
    .undefined,
    .vertex,
    .face,
    .color,
  ]
}

#endif  // swift(>=4.2)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "TensorBoardS.mesh"

extension TensorBoardS_Mesh_MeshPluginData: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".MeshPluginData"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "version"),
    2: .same(proto: "name"),
    3: .standard(proto: "content_type"),
    5: .standard(proto: "json_config"),
    6: .same(proto: "shape"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularInt32Field(value: &self.version)
      case 2: try decoder.decodeSingularStringField(value: &self.name)
      case 3: try decoder.decodeSingularEnumField(value: &self.contentType)
      case 5: try decoder.decodeSingularStringField(value: &self.jsonConfig)
      case 6: try decoder.decodeRepeatedInt32Field(value: &self.shape)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.version != 0 {
      try visitor.visitSingularInt32Field(value: self.version, fieldNumber: 1)
    }
    if !self.name.isEmpty {
      try visitor.visitSingularStringField(value: self.name, fieldNumber: 2)
    }
    if self.contentType != .undefined {
      try visitor.visitSingularEnumField(value: self.contentType, fieldNumber: 3)
    }
    if !self.jsonConfig.isEmpty {
      try visitor.visitSingularStringField(value: self.jsonConfig, fieldNumber: 5)
    }
    if !self.shape.isEmpty {
      try visitor.visitPackedInt32Field(value: self.shape, fieldNumber: 6)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: TensorBoardS_Mesh_MeshPluginData, rhs: TensorBoardS_Mesh_MeshPluginData) -> Bool {
    if lhs.version != rhs.version {return false}
    if lhs.name != rhs.name {return false}
    if lhs.contentType != rhs.contentType {return false}
    if lhs.jsonConfig != rhs.jsonConfig {return false}
    if lhs.shape != rhs.shape {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension TensorBoardS_Mesh_MeshPluginData.ContentType: SwiftProtobuf._ProtoNameProviding {
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "UNDEFINED"),
    1: .same(proto: "VERTEX"),
    2: .same(proto: "FACE"),
    3: .same(proto: "COLOR"),
  ]
}
