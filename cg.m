N= 20000;
h = 1/N;
pts = h:h:1-h;
f = h*h*sinh(pts);
g = [0,0];
f(1) = f(1) - g(1);
f(N-1) = f(N-1) - g(2);
f = f';

e = ones(N-1,1);
A = spdiags([-1*e 2*e -1*e], -1:1, N-1, N-1);

err = 1e-9;
alpha= 1e-4;
MaxIter = 1e8;


tic
CG_Ans =  CG2(A ,f, err);
toc
disp(['Relative Error is ' num2str(100*(norm(CG_Ans - A\f)/norm(A\f))) '%.'])

tic
Matlab_Ans = A\f;
disp(['Elapsed time is ' num2str(toc*1000) ' milliseconds.'])

function x = grad(A,b,tol,alpha, MaxIter)
x = zeros(length(b),1);
r = b;
k = 0;   
tol = tol*norm(b);
    while norm(r) >= tol && k < MaxIter
        r = b- A*x;
        x = x + alpha*r;
        k = k+1;
    end
    norm(r)
end

function x = Steepest_Gradient_Descent(A, b, err, MaxIter)
x = zeros(length(b),1);
r = b;
k = 0;   
err = err*norm(b);
    while norm(r) >= err && k < MaxIter
        r = b- A*x;
        alpha = (r'*r)/(r'*(A*r));
        x = x + alpha*r;
        k = k+1;
    end
end

function x = CG1(A, b, err)
x = zeros(length(b),1);
p = b;
r = b;
err = err*norm(b);
k = 1;
while norm(r) >= err && k < length(b)+1 
    Ap = A*p;
    p2 = p'*Ap;
    alpha = (b'*p)/p2;
    x = x + alpha*p;
    r = r - alpha*Ap;
    beta = - (r'*Ap)/p2;
    p = r + beta*p;
    k = k+1;
end
end

function x = CG2(A, b, err)
x = zeros(length(b),1);
p = b;
r = b;
err = err*norm(b);
k = 1;
r2 = r'*r;
while norm(r) >= err && k < length(b)+1 
    Ap = A*p;
    alpha = r2/(p'*Ap);
    x = x + alpha*p;
    r = r - alpha*Ap;
    prev_r2 = r2;
    r2 = r'*r;
    beta = r2/prev_r2;
    p = r + beta*p;
    k = k+1;
end
end


