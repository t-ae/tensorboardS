import Foundation

enum Summaries {
    static func text(tag: String, text: String) throws -> TensorBoardS_Summary {
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


