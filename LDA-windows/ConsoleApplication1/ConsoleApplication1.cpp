// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "lda.h"

int _tmain(int argc, char* argv[])
{
	int i = 0;
	int n = 11;
	printf("%d\n", n);
	char *paras[11];
	//	-estc -dir D:/wxkdesky/GibbsLDA++-0.2/finalModel -model model-00100 -niters 200 -savestep 100 -twords 10
	while (i<n)
	{
		switch (i)
		{
		case 0:
			paras[i] = "-estc";
			break;
		case 1:
			paras[i] = "-dir";
			break;
		case 2:
			paras[i] = "D:/wxkdesky/GibbsLDA++-0.2/finalModel";
			break;
		case 3:
			paras[i] = "-model";
			break;
		case 4:
			paras[i] = "model-00100";
			break;
		case 5:
			paras[i] = "-niters";
			break;
		case 6:
			paras[i] = "200";
			break;
		case 7:
			paras[i] = "-savestep";
			break;
		case 8:
			paras[i] = "100";
			break;
		case 9:
			paras[i] = "-twords";
			break;
		case 10:
			paras[i] = "10";
			break;
		default:
			break;
		}
		printf("%s\n", paras[i]);
		i++;
	}
	lda_go(n, paras);
	return 0;
}

