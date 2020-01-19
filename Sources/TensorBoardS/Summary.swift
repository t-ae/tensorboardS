import Foundation
import TensorFlow

private func cleanTag(_ name: String) -> String {
    var newName = name
    newName = newName.replacingOccurrences(of: "[^-/\\w.]", with: "_", options: .regularExpression)
    while newName.first == "/" {
        newName = String(newName.dropFirst())
    }
    return newName
}

enum Summaries {
    static func scalar(
        name: String,
        scalar: Float
    ) -> TensorBoardS_Summary {
        let name = cleanTag(name)
        
        return TensorBoardS_Summary.with {
            let value = TensorBoardS_Summary.Value.with {
                $0.tag = name
                $0.simpleValue = scalar
            }
            $0.value = [value]
        }
    }
    
    static func image(
        tag: String,
        image: Tensor<UInt8>
    ) throws -> TensorBoardS_Summary {
        let image = try PNGData(from: image)
        
        return TensorBoardS_Summary.with {
            let value = TensorBoardS_Summary.Value.with {
                $0.tag = tag
                $0.image = TensorBoardS_Summary.Image(png: image)
            }
            $0.value = [value]
        }
    }
    
    static func histogram<Scalar: TensorFlowNumeric>(
        tag: String,
        values: Tensor<Scalar>
    ) -> TensorBoardS_Summary {
        let tag = StringTensor(tag)
        let str = _Raw.histogramSummary(tag: tag, values)
        
        // TODO: str contains protobuf encoded `TensorBoardS_Summary`
        fatalError("Not implemented yet")
    }
    
    static func text(
        tag: String,
        text: String
    ) throws -> TensorBoardS_Summary {
        let content = try TensorBoardS_TextPluginData.with {
            $0.version = 0
        }.serializedData()
        
        return TensorBoardS_Summary.with {
            let value = TensorBoardS_Summary.Value.with {
                $0.tag = tag + "/text_summary"
                
                $0.metadata = TensorBoardS_SummaryMetadata.with {
                    $0.pluginData = TensorBoardS_SummaryMetadata.PluginData.with {
                        $0.pluginName = "text"
                        $0.content = content
                    }
                }
                
                $0.tensor = TensorBoardS_TensorProto.with {
                    $0.dtype = .dtString
                    $0.stringVal = [text.data(using: .utf8)!]
                    $0.tensorShape = TensorBoardS_TensorShapeProto()
                    $0.tensorShape.dim = [1]
                }
            }
            $0.value = [value]
        }
    }
}


