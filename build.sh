g++ -c audio.cpp -I ./SFML-3.0.0/include
g++ audio.o -o audio -L ./SFML-3.0.0/lib -lsfml-audio -lsfml-system
export LD_LIBRARY_PATH=./SFML-3.0.0/lib && ./audio
