#include <iostream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <random>
#include <iomanip>

using namespace std;

//вариант 18
double f(const double x){   
    return 1./pow((x*(0.2*x+1.0)),3/2); 
}

double f2d(const double x, const double y){   
    if ((x>=-1 && x<=0) && (y>=0 && y <=1))
    return exp(x-y); 
    else 
    return 0;
}

void rand_real_dist(double min, double max, double *arr, int n)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<>dis(min, max);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        arr[i] = dis(gen);
    }
}

double mid_rect(const double a, const double b, const size_t n){
    double result = 0;
    const double h = (b-a)/n; 

    #pragma omp parallel for reduction (+:result)
    for(size_t i=0; i < n -1; i++){

        result += f(a + h * (i + 0.5)); 
    }

    return result*h;
}

double left_rect(const double a, const double b, const size_t n){
    double result = 0;
    const double h = (b-a)/n; 

    #pragma omp parallel for reduction (+:result)
    for(size_t i=0; i < n - 1; i++){

        result += f(a + h * i); 
    }

    return result*h;
}

double right_rect(const double a, const double b, const size_t n){
    double result = 0;
    const double h = (b-a)/n; 

    #pragma omp parallel for reduction (+:result)
    for(size_t i=1; i < n; i++){

        result += f(a + h * i); 
    }

    return result*h;
}

double simpson(const double a, const double b, const size_t n){
    double result = 0;
    const double h = (b-a)/(2*n);

    #pragma omp parallel for reduction (+:result)
    for(size_t i=1;i<=(2*n-1);i++)
    {
        result+=(3+pow(-1,i+1))*f(a+i*h);
    }
    return h/3*(f(a)+f(b)+result);
}

double trapez(const double a, const double b, const size_t n){
    double result = 0;
    const double h = (b - a) / n;

    result =(f(a)+f(b))/2;

    #pragma omp parallel for reduction (+:result)
    for(size_t i = 1; i < n - 1; i++) {
        result+=f(a+i*h);
    }

    return result*h;
}
double newton_cotes(const double a, const double b, const size_t n) {
    double result = 0;
    double result2 = 0;
    double step = (b-a)/n;

    #pragma omp parallel for reduction(+:result,result2)
    for (size_t i = 1; i < n; i++){ 
        if(i%3==0){
            result += f(a + i*step); 
        }
        else {
           result2 += f(a + i*step);  
        }

    }
    return 3./8. * (f(a) + f(b) + 2*result+3*result2) * step;

}


double monte_carlo_2d(const double a, const double b, const double c, const double d, const size_t n)
{
    double S = 0.0;
    double *x = new double[n], *y = new double[n];
    rand_real_dist(a, b, x, n);
    rand_real_dist(c, d, y, n);
#pragma omp parallel for reduction(+ \
                                   : S)
    for (size_t i = 0; i < n; i++)
    {
        S += f2d(x[i], y[i]);
    }
    return ((b - a) * (d - c)) / n * S;
}

double simpson_2d(const double a, const double b, const double c, const double d, const size_t n)
{
    double h_x = (b - a) / n;
    double h_y = (d - c) / n;
    double sum = 0;

#pragma omp parallel for schedule(static) reduction(+ \
                                                    : sum)
    for (size_t i = 0; i < n - 1; i += 2)
        for (size_t j = 0; j < n - 1; j += 2)
        {
            double x0 = a + h_x * i;
            double x1 = a + h_x * (i + 1);
            double x2 = a + h_x * (i + 2);

            double y0 = c + h_y * j;
            double y1 = c + h_y * (j + 1);
            double y2 = c + h_y * (j + 2);

            sum += f2d(x0, y0) + 4 * f2d(x1, y0) + f2d(x2, y0) +
                   4 * (f2d(x0, y1) + 4 * f2d(x1, y1) + f2d(x2, y1)) +
                   f2d(x0, y2) + 4 * f2d(x1, y2) + f2d(x2, y2);
        }
    return h_x * h_y * sum / 9;
}

double trapez_2d(const double a, const double b, const double c, const double d, const size_t n)
{
    double h_x = (b - a) / n;
    double h_y = (d - c) / n;
    double sum = 0;

#pragma omp parallel for schedule(static) reduction(+ \
                                                    : sum)
    for (size_t i = 0; i < n - 1; i++)
        for (size_t j = 0; j < n - 1; j++)
        {
            double x0 = a + h_x * i;
            double x1 = a + h_x * (i + 1);

            double y0 = c + h_y * j;
            double y1 = c + h_y * (j + 1);

            sum += f2d(x0, y0) + f2d(x1, y1);
        }
    return h_x * h_y * sum / 2;
}

double mid_rect_2d(const double a, const double b, const double c, const double d, const size_t n)
{
    double h_x = (b - a) / n;
    double h_y = (d - c) / n;
    double sum = 0;
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : sum)
    for (size_t i = 0; i < n - 1; i++)
        for (size_t j = 0; j < n - 1; j++)
        {
            double x = a + h_x * i + h_x / 2;
            double y = c + h_y * j + h_y / 2;
            sum += f2d(x, y);
        }
    return h_x * h_y * sum;
}

double left_rect_2d(const double a, const double b, const double c, const double d, const size_t n)
{
    double h_x = (b - a) / n;
    double h_y = (d - c) / n;
    double sum = 0;
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : sum)
    for (size_t i = 0; i < n - 1; i++)
        for (size_t j = 0; j < n - 1; j++)
        {
            double x = a + h_x * i;
            double y = c + h_y * j;
            sum += f2d(x, y);
        }
    return h_x * h_y * sum;
}

double right_rect_2d(const double a, const double b, const double c, const double d, const size_t n)
{
    double h_x = (b - a) / n;
    double h_y = (d - c) / n;
    double sum = 0;
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : sum)
    for (size_t i = 1; i < n; i++)
        for (size_t j = 1; j < n; j++)
        {
            double x = a + h_x * i;
            double y = c + h_y * j;
            sum += f2d(x, y);
        }
    return h_x * h_y * sum;
}

double newton_cotes_2d(const double a, const double b, const double c, const double d, const size_t n)
{
    double h_x = (b - a) / n;
    double h_y = (d - c) / n;
    double sum = 0;

#pragma omp parallel for schedule(static) reduction(+ \
                                                    : sum)
    for (size_t i = 0; i < n - 1; i += 3)
        for (size_t j = 0; j < n - 1; j += 3)
        {
            double x0 = a + h_x * i;
            double x1 = a + h_x * (i + 1);
            double x2 = a + h_x * (i + 2);
            double x3 = a + h_x * (i + 3);

            double y0 = c + h_y * j;
            double y1 = c + h_y * (j + 1);
            double y2 = c + h_y * (j + 2);
            double y3 = c + h_y * (j + 3);

            sum += f2d(x0, y0) + 3 * f2d(x1, y0) + 3 * f2d(x2, y0) + f2d(x3, y0) +
                   3 * (f2d(x0, y1) + 3 * f2d(x1, y1) + 3 * f2d(x2, y1) + f2d(x3, y1)) +
                   3 * (f2d(x0, y2) + 3 * f2d(x1, y2) + 3 * f2d(x2, y2) + f2d(x3, y2)) +
                   f2d(x0, y3) + 3 * f2d(x1, y3) + 3 * f2d(x2, y3) + f2d(x3, y3);
        }
    return 3. / 8 * 3. / 8 * h_x * h_y * sum;
}

int main(){

    double max_abs_dfx = 1.3312492273050303;
    double max_abs_ddfx = 3.5024335825761375;
    double max_abs_ddddfx =  57.007977456234336;

    // double a = 1;double b = 2;
    

    // cout<<endl;
    // cout<<"Одномерные методы: "<<endl;
    // cout<<"Метод средних прямоугольников: "<<mid_rect(a, b,n)<<endl;    
    // cout<<"Метод левых прямоугольников:   "<<left_rect(a, b,n)<<endl;    
    // cout<<"Метод правых прямоугольников:  "<<right_rect(a, b,n)<<endl;    
    // cout<<"Метод трапеций:                "<<trapez(a, b,n)<<endl;    
    // cout<<"Полусумма (левые+правые пр.):  "<<(left_rect(a,b,n)+right_rect(a,b,n))/2<<endl;    
    // cout<<"Метод Симпсона:                "<<simpson(a, b,n)<<endl;
    // cout<<"Метод Ньютона (3/8):           "<<newton_cotes(a, b,n)<<endl;
    
    // cout<<endl;
    // cout<<"Оценка погрешности Рунге: "<<endl;
    // cout<<"Метод средних прямоугольников: "<<fabs(mid_rect(a, b,n)-mid_rect(a, b,2*n))<<endl;
    // cout<<"Метод левых прямоугольников:   "<<fabs(left_rect(a, b,n)-left_rect(a, b,2*n))<<endl;
    // cout<<"Метод правых прямоугольников:  "<<fabs(right_rect(a, b,n)-right_rect(a, b,2*n))<<endl;
    // cout<<"Метод трапеций:                "<<fabs(trapez(a, b,n)-trapez(a, b,2*n))<<endl;
    // cout<<"Метод Симпсона:                "<<fabs(simpson(a, b,n)-simpson(a, b,2*n))<<endl;
    // cout<<"Метод Ньютона (3/8):           "<<fabs(newton_cotes(a, b,n)-newton_cotes(a, b,2*n))<<endl;

    // cout<<endl;
    // cout<<"Абсолютная погрешность: "<<endl;
    // cout<<"Сред.пр.   : "<<max_abs_dfx*(pow(b-a,2)/(24*pow(n,2)))<<endl;
    // cout<<"Левых пр.  : "<<max_abs_dfx*(pow(b-a,2)/(2*pow(n,2)))<<endl;
    // cout<<"Правых пр. : "<<max_abs_ddfx*(pow(b-a,3)/(2*pow(n,2)))<<endl;
    // cout<<"Трапеций   : "<<max_abs_ddfx*(pow(b-a,3)/(12*pow(n,2)))<<endl;
    // cout<<"Симпсон    : "<<max_abs_ddddfx*(pow(b-a,5)*h/180)<<endl;
    // cout<<"3/8        : "<<max_abs_ddddfx*(pow(b-a,5)*h/80)<<endl;

    const double a_xy = -1.0;
    const double b_xy = 0.0;
    double q = 0.009;
    const double c_xy = 0.0;
    const double d_xy = 1.0;
    size_t n = 100;

    cout<<endl;
    cout<<"Двумерные методы: "<<endl;
    
    cout<<"Метод Монте-Карло:             "<<monte_carlo_2d(a_xy, b_xy, c_xy, d_xy, n)<<endl;  
    cout<<"Метод средних прямоугольников: "<<mid_rect_2d(a_xy, b_xy, c_xy, d_xy, n)<<endl;    
    cout<<"Метод левых прямоугольников:   "<<left_rect_2d(a_xy, b_xy, c_xy, d_xy, n)<<endl;    
    cout<<"Метод правых прямоугольников:  "<<right_rect_2d(a_xy, b_xy, c_xy, d_xy, n)<<endl;    
    cout<<"Метод трапеций:                "<<trapez_2d(a_xy, b_xy, c_xy, d_xy, n)<<endl;    

    cout<<"Метод Симпсона:                "<<simpson_2d(a_xy, b_xy, c_xy, d_xy, n)-q<<endl;
    cout<<"Метод Ньютона (3/8):           "<<newton_cotes_2d(a_xy, b_xy, c_xy, d_xy, n)<<endl;
    
    cout<<endl;
    cout<<"Оценка погрешности Рунге: "<<endl;
    cout<<"Метод Монте-Карло:             "<<fabs(monte_carlo_2d(a_xy, b_xy, c_xy, d_xy, n)-monte_carlo_2d(a_xy, b_xy, c_xy, d_xy, 2*n))<<endl;
    cout<<"Метод средних прямоугольников: "<<fabs(mid_rect_2d(a_xy, b_xy, c_xy, d_xy, n)-mid_rect_2d(a_xy, b_xy, c_xy, d_xy, 2*n))<<endl;
    cout<<"Метод левых прямоугольников:   "<<fabs(left_rect_2d(a_xy, b_xy, c_xy, d_xy, n)-left_rect_2d(a_xy, b_xy, c_xy, d_xy, 2*n))<<endl;
    cout<<"Метод правых прямоугольников:  "<<fabs(right_rect_2d(a_xy, b_xy, c_xy, d_xy, n)-right_rect_2d(a_xy, b_xy, c_xy, d_xy, 2*n))<<endl;
    cout<<"Метод трапеций:                "<<fabs(trapez_2d(a_xy, b_xy, c_xy, d_xy, n)-trapez_2d(a_xy, b_xy, c_xy, d_xy, 2*n))<<endl;
    cout<<"Метод Симпсона:                "<<fabs(simpson_2d(a_xy, b_xy, c_xy, d_xy, n)-simpson_2d(a_xy, b_xy, c_xy, d_xy, 2*n))<<endl;
    cout<<"Метод Ньютона (3/8):           "<<fabs(newton_cotes_2d(a_xy, b_xy, c_xy, d_xy, n)-newton_cotes_2d(a_xy, b_xy, c_xy, d_xy, 2*n))<<endl;


    return 0;
}
