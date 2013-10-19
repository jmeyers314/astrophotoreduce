import pgm
import pipes
import tempfile

def read_CR2_as_CFA(filename):
    """ Read Canon T3i RAW file directly into a numpy ndarray, using a tempfile and some UNIX piping.
    """
    p = pipes.Template()
    p.append('dcraw -D -4 -j -t 0 -c $IN', 'f-')
    t = tempfile.NamedTemporaryFile('r')
    p.copy(filename, t.name)
    cfa=pgm.read_pgm(t.name)
    return cfa
