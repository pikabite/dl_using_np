docker run --runtime=nvidia -v ${PWD}:/dy -d -it --name ann_with_np -p 12005:12005 -p 6101:6101 ann_with_np:1
