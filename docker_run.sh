docker run --runtime=nvidia -v ${PWD}:/workspace -d -it --name dl_with_np -p 15015:15015 -p 6007:6007 dl_using_np:1
