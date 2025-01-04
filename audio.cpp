#include <SFML/Audio.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
	if (argc < 7) {
		return -1;
    }

    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile(argv[1])) {
		return -1;
    }
	
	sf::Sound sound(buffer);
	float volume = std::stof(argv[2])*10;
	float pitch = std::stof(argv[3]);
	float x = std::stof(argv[4]);
	float y = std::stof(argv[5]);
	float z = std::stof(argv[6]);
    sound.setVolume(volume);
    sound.setPitch(pitch);
    sound.setPosition({x, y, z});
    sound.play();

	while (sound.getStatus() == sf::Sound::Status::Playing) {sf::sleep(sf::milliseconds(100));  }

    return 0;
	
}
