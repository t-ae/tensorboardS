import Foundation
import CryptoSwift

class RecordWriter {
    let fileURL: URL
    private let fileHandle: FileHandle
    
    init(fileURL: URL) throws {
        self.fileURL = fileURL
        
        let parent = fileURL.deletingLastPathComponent()
        let fm = FileManager.default
        try? fm.createDirectory(at: parent, withIntermediateDirectories: true, attributes: nil)
        if !fm.fileExists(atPath: fileURL.path) {
            fm.createFile(atPath: fileURL.path, contents: nil, attributes: nil)
        }
        self.fileHandle = try FileHandle(forWritingTo: fileURL)
    }
    
    func write(_ data: Data) {
        var len = UInt64(data.count)
        let header = Data(bytes: &len, count: MemoryLayout<UInt64>.size)
        fileHandle.write(header)
        fileHandle.write(maskedCRC32C(header))
        fileHandle.write(data)
        fileHandle.write(maskedCRC32C(data))
    }
    
    func flush() {
        fileHandle.synchronizeFile()
    }
    
    func close() {
        flush()
        fileHandle.closeFile()
    }
}

private func maskedCRC32C(_ data: Data) -> UInt32 {
    let checksum = Checksum.crc32c([UInt8](data))
    let x15 = checksum >> 15
    let x17 = UInt32(truncatingIfNeeded: UInt64(checksum) << 17)
    return UInt32(truncatingIfNeeded: UInt64(x15 | x17) + 0xa282ead8)
}

extension FileHandle {
    fileprivate func write<T: UnsignedInteger>(_ value: T) {
        var value = value
        write(Data(bytes: &value, count: MemoryLayout<T>.size))
    }
}