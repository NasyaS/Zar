#include <iostream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <cmath>

using namespace std;

//вариант 18
double f(const double x){   
    return 1./pow((x*(0.2*x+1.0)),3/2); 
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


int main(){

    double max_abs_dfx = 1.3312492273050303;
    double max_abs_ddfx = 3.5024335825761375;
    double max_abs_ddddfx =  57.007977456234336;

    double a = 1;double b = 2;size_t n = 10000;double h = (b-a)/n;

    cout<<endl;
    cout<<"Методы: "<<endl;
    cout<<"Метод средних прямоугольников: "<<mid_rect(a, b,n)<<endl;    
    cout<<"Метод левых прямоугольников:   "<<left_rect(a, b,n)<<endl;    
    cout<<"Метод правых прямоугольников:  "<<right_rect(a, b,n)<<endl;    
    cout<<"Метод трапеций:                "<<trapez(a, b,n)<<endl;    
    cout<<"Полусумма (левые+правые пр.):  "<<(left_rect(a,b,n)+right_rect(a,b,n))/2<<endl;    
    cout<<"Метод Симпсона:                "<<simpson(a, b,n)<<endl;
    cout<<"Метод Ньютона (3/8):           "<<newton_cotes(a, b,n)<<endl;
    
    cout<<endl;
    cout<<"Оценка погрешности Рунге: "<<endl;
    cout<<"Метод средних прямоугольников: "<<fabs(mid_rect(a, b,n)-mid_rect(a, b,2*n))<<endl;
    cout<<"Метод левых прямоугольников:   "<<fabs(left_rect(a, b,n)-left_rect(a, b,2*n))<<endl;
    cout<<"Метод правых прямоугольников:  "<<fabs(right_rect(a, b,n)-right_rect(a, b,2*n))<<endl;
    cout<<"Метод трапеций:                "<<fabs(trapez(a, b,n)-trapez(a, b,2*n))<<endl;
    cout<<"Метод Симпсона:                "<<fabs(simpson(a, b,n)-simpson(a, b,2*n))<<endl;
    cout<<"Метод Ньютона (3/8):           "<<fabs(newton_cotes(a, b,n)-newton_cotes(a, b,2*n))<<endl;

    cout<<endl;
    cout<<"Абсолютная погрешность: "<<endl;
    cout<<"Сред.пр.   : "<<max_abs_dfx*(pow(b-a,2)/(24*pow(n,2)))<<endl;
    cout<<"Левых пр.  : "<<max_abs_dfx*(pow(b-a,2)/(2*pow(n,2)))<<endl;
    cout<<"Правых пр. : "<<max_abs_ddfx*(pow(b-a,3)/(2*pow(n,2)))<<endl;
    cout<<"Трапеций   : "<<max_abs_ddfx*(pow(b-a,3)/(12*pow(n,2)))<<endl;
    cout<<"Симпсон    : "<<max_abs_ddddfx*(pow(b-a,5)*h/180)<<endl;
    cout<<"3/8        : "<<max_abs_ddddfx*(pow(b-a,5)*h/80)<<endl;

    return 0;
}
