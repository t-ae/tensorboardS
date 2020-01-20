import Foundation

public struct GenericError: Error {
    public let message: String
    
    init(_ message: String) {
        self.message = message
    }
}
