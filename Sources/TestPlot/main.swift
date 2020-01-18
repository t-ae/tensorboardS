import Foundation
import TensorBoardS

let logdir = URL(fileURLWithPath: "./logdir")
let writer = try SummaryWriter(logdir: logdir)

writer.addText(tag: "test_tag", text: "hoge", date: Date(timeIntervalSince1970: 0))

// MARK: - Add scalars to draw graph

for i in 0..<100 {
    let sin = sinf(Float(i) / 30)
    let cos = cosf(Float(i) / 30)
    writer.addScalar(tag: "sin", scalar: sin, step: i)
    writer.addScalar(tag: "cos", scalar: cos, step: i)
}

writer.flush()
writer.close()
