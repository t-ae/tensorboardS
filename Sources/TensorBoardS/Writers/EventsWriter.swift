import Foundation

class EventsWriter {
    let writer: RecordWriter
    let lock = NSLock()
    
    private var closed = false
    
    init(logdir: URL, filenamePrefix: String, filenameSuffix: String = "") throws {
        let unixtime = Date().timeIntervalSince1970
        
        let timePart = String(unixtime).prefix(10)
        let hostName = Host.current().name ?? ""
        let filename = """
        \(filenamePrefix).out.tfevents.\(timePart).\(hostName)\(filenameSuffix)
        """
        
        let fileURL = logdir.appendingPathComponent(filename)
        writer = try RecordWriter(fileURL: fileURL)
        
        let event = TensorBoardS_Event.with {
            $0.wallTime = Date().timeIntervalSince1970
            $0.fileVersion = "brain.Event:2"
        }
        write(event: event)
    }
    
    func write(event: TensorBoardS_Event) {
        lock.lock()
        defer { lock.unlock() }
        
        do {
            let data = try event.serializedData()
            writer.write(data)
        } catch {
            print("[EventsWriter] Failed to write: \(error)")
        }
    }
    
    func flush() {
        lock.lock()
        defer { lock.unlock() }
        
        writer.flush()
    }
    
    func close() {
        lock.lock()
        defer { lock.unlock() }
        
        writer.close()
    }
}
