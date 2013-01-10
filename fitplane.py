import numpy
import pylab
def fitplane(x,y,z):
    """
    returns a least-square fitted slope using (fast) matrix method.
    """
    xx=numpy.ravel(x)
    yy=numpy.ravel(y)
    fit=numpy.ravel(z)

    X=numpy.matrix([numpy.ones(len(fit)),xx,yy])


    X=X.T
    fit=numpy.matrix(fit)
    fit=fit.T

    beta=numpy.dot(((numpy.dot(X.T,X)).I),numpy.dot(X.T,fit))

    return numpy.reshape(numpy.dot(X,beta),z.shape)

if __name__=="__main__":
  
    X,Y=numpy.meshgrid(numpy.arange(10),numpy.arange(10))
    pylab.imshow(X+Y)
    pylab.title("input")
    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.colorbar()
    pylab.show()
    pylab.close()
    pylab.imshow(fitplane(X,Y,X+Y))
    pylab.title("reconstructed")
    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.colorbar()
    pylab.show()
    pylab.close()
    pylab.imshow(X+Y-fitplane(X,Y,X+Y))
    pylab.title("difference")
    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.colorbar()
    pylab.show()
    pylab.close()
    
  
  
