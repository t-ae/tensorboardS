import Foundation
import TensorFlow
import TensorBoardS

let logdir = URL(fileURLWithPath: "./logdir")
let writer = try SummaryWriter(logdir: logdir)
writer.errorHandler = { error in
    print(error)
}

// MARK: - Add text
writer.addText(tag: "test_tag", text: "hoge", date: Date(timeIntervalSince1970: 0))

// MARK: - Add scalars to draw graph

for i in 0..<100 {
    let sin = sinf(Float(i) / 30)
    let cos = cosf(Float(i) / 30)
    writer.addScalar(tag: "sin", scalar: sin, step: i)
    writer.addScalar(tag: "cos", scalar: cos, step: i)
}

// MARK: - Add image

var gray = Tensor<Float>((0..<64).map { Float($0) / 64 }).reshaped(to: [8, 8])
gray[0, 0] = Tensor(0)
gray[0, 7] = Tensor(0)
gray[7, 0] = Tensor(0)
gray[7, 7] = Tensor(0)
var rgb = Tensor<Float>(randomUniform: [8, 8, 3])
rgb[0, 0] = Tensor([1, 0, 0])
rgb[0, 7] = Tensor([0, 1, 0])
rgb[7, 0] = Tensor([0, 0, 1])
rgb[7, 7] = Tensor([0, 0, 0])

writer.addImage(tag: "gray", image: gray)
writer.addImage(tag: "rgb", image: rgb)

writer.flush()
writer.close()
