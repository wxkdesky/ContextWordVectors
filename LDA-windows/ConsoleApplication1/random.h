int seed;

void srandom(int s)
{
	seed = s;
}

int random()
{
	seed = seed * 22695477 + 1;
	return (seed >> 16) & 0x7fff;
}