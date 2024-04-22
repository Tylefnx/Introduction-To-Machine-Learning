#include<stdio.h>

int power(int x, int y){
    if(y == 0)
        return 1;
    else if(y % 2 == 1)
        return x * power(x, y-1); // Burhan hoca cozum => return power(x, y/2) * power(x,y/2) * x;
    else
        return power(x, (y/2)) * power(x, (y/2));
}

int main(){
    printf("%d", power(2,5));
    return 0;
}
