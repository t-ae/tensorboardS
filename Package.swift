// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorBoardS",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "TensorBoardS",
            targets: ["TensorBoardS"]),
        .executable(
            name: "TestPlot",
            targets: ["TestPlot"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.7.0"),
        .package(url: "https://github.com/krzyzanowskim/CryptoSwift.git", from: "1.3.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "TensorBoardS",
            dependencies: ["SwiftProtobuf", "CryptoSwift"]),
        .target(
            name: "TestPlot",
            dependencies: ["TensorBoardS"]),
        .testTarget(
            name: "TensorBoardSTests",
            dependencies: ["TensorBoardS"]),
    ]
)
