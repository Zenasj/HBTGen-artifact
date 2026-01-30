import tensorflow as tf

class Flow( tfb.Bijector ):

    def __init__( self, theta, a, **kwargs ):
        tfb.Bijector.__init__( self, forward_min_event_ndims = 0, **kwargs )
        bijectors = [ tfb.Tanh() ]
        self.chain = tfb.Chain( bijectors = bijectors )

    def _forward( self, z ):
        return self.chain( z )

    def _inverse( self, x ):
        result = self.chain.inverse( x ) 
        return result

    def _forward_log_det_jacobian( self, z ):
        return self.chain._forward_log_det_jacobian( z, event_ndims = 2 )

Z = tf.convert_to_tensor( [ [ [ 0.1, 0.2 ], [ 0.3, 0.4 ], [ 0.5, 0.6 ] ], 
                            [ [ 0.8, 0.7 ], [ 0.6, 0.5 ], [ 0.4, 0.3 ] ],
                            [ [ 0.4, 0.7 ], [ 0.2, 0.1 ], [ 0.8, 0.0 ] ] ] )
print( "Z", Z )
nf = Flow( 1., 2. )  # ### theta, a 
bd = tfd.MultivariateNormalDiag( loc=[0.,0.], scale_diag=[1.,1.] )
td = tfd.TransformedDistribution( bd, nf )
td.log_prob( Z )