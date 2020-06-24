docker run --runtime=nvidia -v ${PWD}:/dy -d -it --name dy_dl_with_np -p 15015:15015 -p 6007:6007 dy_dl_with_np:1
