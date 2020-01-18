import XCTest
@testable import TensorBoardS

final class TensorBoardSTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(TensorBoardS().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
