build:
    docker build -t dhondta/packing-box .

clean:
    docker ps -a --filter "ancestor=dhondta/packing-box" -q | xargs -r docker rm -f

run:
    docker run -it -h packing-box -v `pwd`:/mnt/share dhondta/packing-box
