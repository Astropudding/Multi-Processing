#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;

double eps;
double A [N][N][N];

void relax();
void init();
void verify();

int main(int an, char **as)
{
    int it;
    
    init();
    double start = omp_get_wtime();
    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax();
        printf( "it=%4i   eps=%f\n", it,eps);
        if (eps < maxeps) break;
    }
    
    verify();
    double finish = omp_get_wtime();
    printf("time = %f\n", finish - start);
    return 0;
}


void init()
{
#pragma omp parallel for shared (A) private (i, j, k)
    for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
            for(k=0; k<=N-1; k++)
            {
                if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
                    A[i][j][k]= 0.;
                else A[i][j][k]= ( 4. + i + j + k) ;
            }
    
}
void relax()
{
#pragma omp parallel for shared (A) private (i, j, k)
    for(j=1; j<=N-2; j++)
        for(i=1; i<=N-2; i++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
            }
#pragma omp parallel for shared (A) private (i, j, k)
    for(i=1; i<=N-2; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
            }
#pragma omp parallel for shared (A) private (i, j, k)
    for(i=1; i<=N-2; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                double e;
                e=A[i][j][k];
                A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
                e = fabs(e-A[i][j][k]);
                if (eps < e){
                      #pragma omp critical
                      {
                          eps = e;
                      }
                }
            }
    
}

void verify()
{
    double s;
    
    s=0.;
#pragma omp parallel for shared (A) private (i, j, k) reduction(+:s)
    for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
            for(k=0; k<=N-1; k++)
            {
                s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
            }
    printf("  S = %f\n",s);
    
}
