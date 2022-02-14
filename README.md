# CSIRO Runner

## Usage

Terminal 1

```sh
$ python poller.py path/to/directory
```

Terminal 2 (On the same machine)

```sh
$ python serving.py
```

To update the service proto python files, run

```sh
python -m grpc_tools.protoc -I.  --python_out=. --grpc_python_out=. service.proto
```

## Poller

The poller watches for the `*.jpg` file additions in the specified directory.
When new files are added, it will add them to dataset and fetch an image per
second. This should be improved to match up with realtime performance
requirement.

## Detector

The detector runs inference, and maintains gRPC endpoint to accept inference
requests and returning detection results. `service.proto` defines the gRPC
service.

## Benchmark

Simply measure the model's performance.

```sh
python benchmark.py \
  --model_path=path/to/model --image_path=path/to/image/dir --batch_size=1
```

## TODO

*   Load management between poller and detector to achieve > 10fps performance.
*   Dockerize (taeheej@)
*   Tracker integration (swatisingh@)
*   Performance optimizations (cheshire@)
