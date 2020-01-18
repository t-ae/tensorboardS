import Foundation
import TensorBoardS

let logdir = URL(fileURLWithPath: "./logdir")
let writer = try SummaryWriter(logdir: logdir)

writer.addText(tag: "test_tag", text: "hoge", date: Date(timeIntervalSince1970: 0))
writer.flush()
writer.close()
