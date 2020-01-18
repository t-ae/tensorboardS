#!/bin/sh

rm Sources/TensorBoardS/proto/*.swift 2> /dev/null 

protoc --swift_out=Sources/TensorBoardS proto/*.proto
