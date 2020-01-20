import Foundation

public struct TensorBoardSError: Error {
    public let message: String
    
    init(_ message: String) {
        self.message = message
    }
}
