g++ -c main.cpp -I ./SFML-3.0.0/include
g++ main.o -o main -L ./SFML-3.0.0/lib -lsfml-audio -lsfml-system
export LD_LIBRARY_PATH=./SFML-3.0.0/lib && ./main

