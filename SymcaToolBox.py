import subprocess
from os import devnull, path, mkdir
import sys
from re import sub
from sympy import Symbol, sympify, nsimplify, fraction, S
from sympy.matrices import Matrix, diag, NonSquareMatrixError
from CCobjects import CCBase, CCoef
import logging


class SymcaToolBox(object):
    """The class with the functions used to populate SymcaData. The project is
    structured in this way to abstract the 'work' needed to build the various
    matrices away from the SymcaData class. This 'toolbox' does only has
    a filename variable used for temp storage of maxima output"""


    # @staticmethod
    # def make_path(mod,subdir,subsubdir = None):
    #     base_dir = mod.ModelOutput
    #     main_dir = base_dir + '/' + subdir
    #     mod_dir = main_dir + '/' + mod.ModelFile[:-4]
    #     if not path.exists(main_dir):
    #         mkdir(main_dir)
    #     if not path.exists(mod_dir):
    #         mkdir(mod_dir)
    #     if subsubdir:
    #         branch_dir = mod_dir + '/' + subsubdir
    #         if not path.exists(branch_dir):
    #             mkdir(branch_dir)
    #         return branch_dir + '/'
        


    @staticmethod
    def get_nmatrix(mod):
        """
        Returns a sympy matrix made from the N matrix in a Pysces model where
        the elements are in the same order as they appear in the k and l
        matrices in pysces.

        We need this to make calculations easier later on.
        """
        nmatrix = mod.nmatrix
        #swap columns around to same order as kmatrix, store in new matrix
        nmatrix_cols = nmatrix[:, mod.kmatrix_row]
        #swap rows around to same oder as lmatrix, store in a new matrix
        nmatrix_cols_rows = nmatrix_cols[mod.lmatrix_row, :]
        #create Sympy symbolic matrix from the numpy ndarray
        nmat = Matrix(nmatrix_cols_rows)
        return nmat

    @staticmethod
    def get_num_ind_species(mod):
        inds = len(mod.lmatrix_col)
        return inds

    @staticmethod
    def get_num_ind_fluxes(mod):
        inds = len(mod.kmatrix_col)
        return inds

    @staticmethod
    def get_species_vector(mod):
        """
        Returns a vector (sympy matrix) with the species in the correct order
        """
        slist = []
        #gets the order of the species from the lmatrix rows
        for index in mod.lmatrix_row:
            slist.append(mod.species[index])

        svector = Matrix(sympify(slist))
        #inds = len(mod.lmatrix_col)
        #Sind = Matrix(svector[:inds])
        #Sdep = Matrix(svector[inds:])
        return svector

    @staticmethod
    def get_fluxes_vector(mod):
        """
        Gets the dependent and independent fluxes (in the correct order)
        """

        jlist = []
        #gets the order of the fluxes from the kmatrix rows
        for index in mod.kmatrix_row:
            jlist.append('J_' + mod.reactions[index])
        jvector = Matrix(sympify(jlist))
        #inds = len(mod.kmatrix_col)
        #Jind = Matrix(jvector[:inds])
        #Jdep = Matrix(jvector[inds:])
        return jvector

    @staticmethod
    def substitute_fluxes(all_fluxes, kmatrix):
        """
        Substitutes equivalent fluxes in the kmatrix (e.i. dependent fluxes
        with independent fluxes or otherwise equal fluxes)
        """
        new_fluxes = all_fluxes[:, :]
        for row in xrange(kmatrix.rows - 1, -1, -1):
            for row_above in xrange(row - 1, -1, -1):
                if kmatrix[row, :] == kmatrix[row_above, :]:
                    new_fluxes[row] = new_fluxes[row_above]
        return new_fluxes

    @staticmethod
    def scale_matrix(all_elements, mat, inds):
        """
        Scales the k or l matrix.

        The procedure is the same for each matrix:
           (D^x)^(-1)   *          y         *        D^(x_i)

        Inverse diagonal   The matrix to be      The diagonal of
        of the x where     scaled. i.e. the      the independent x
        x is either the    k or l matrix         where x is the
        species or the                           species or the
        fluxes                                   fluxes

        """
        d_all_inv = diag(*all_elements).inv()
        d_inds = diag(*inds)
        scaled_matrix = d_all_inv * mat * d_inds
        return scaled_matrix

    @staticmethod
    def get_es_matrix(mod, nmatrix, fluxes, species):
        """
        Gets the esmatrix.

        Goes down the columns of the nmatrix (which holds the fluxes)
        to get the rows of the esmatrix.

        Nested loop goes down the rows of the nmatrix (which holds the species)
        to get the columns of the esmatrix

        so the format is

        ecReationN0_M0 ecReationN0_M1 ecReationN0_M2
        ecReationN1_M0 ecReationN1_M1 ecReationN1_M2
        ecReationN2_M0 ecReationN2_M1 ecReationN2_M2
        """
        nmat = nmatrix

        elas = []

        for col in range(nmat.cols):
            current_reaction = fluxes[col]
            elas_row = []
            for row in range(nmat.rows):
                current_species = species[row]
                ec_name = 'ec' + str(current_reaction)[2:] + '_' + str(current_species)
                cond1 = getattr(mod, ec_name) != 0

                if cond1:
                    elas_row.append(ec_name)
                else:
                    elas_row.append(0)
            elas.append(elas_row)

        esmatrix = Matrix(elas)
        return esmatrix

    @staticmethod
    def simplify_matrix(matrix):
        """
        Replaces floats with ints and puts elements with fractions
        on a single demoninator.
        """
        m = matrix[:, :]
        for i, e in enumerate(m):
            m[i] = nsimplify(e, rational=True).cancel()
        return m

    @staticmethod
    def adjugate_matrix(matrix):
        """
        Returns the adjugate matrix which is the transpose of the
        cofactor matrix.

        Contains code adapted from sympy.
        Specifically:

        cofactorMatrix()
        minorEntry()
        minorMatrix()
        cofactor()
        """


        def cofactor_matrix(mat):
            out = Matrix(mat.rows, mat.cols, lambda i, j:
            cofactor(mat, i, j))
            return out

        def minor_entry(mat, i, j):
            if not 0 <= i < mat.rows or not 0 <= j < mat.cols:
                raise ValueError("`i` and `j` must satisfy 0 <= i < `mat.rows` " +
                                 "(%d)" % mat.rows + "and 0 <= j < `mat.cols` (%d)." % mat.cols)
            return SymcaToolBox.det_bareis(minor_matrix(mat, i, j))

        def minor_matrix(mat, i, j):
            if not 0 <= i < mat.rows or not 0 <= j < mat.cols:
                raise ValueError("`i` and `j` must satisfy 0 <= i < `mat.rows` " +
                                 "(%d)" % mat.rows + "and 0 <= j < `mat.cols` (%d)." % mat.cols)
            m = mat.as_mutable()
            m.row_del(i)
            m.col_del(j)
            return m[:, :]


        def cofactor(mat, i, j):
            if (i + j) % 2 == 0:
                return minor_entry(mat, i, j)
            else:
                return -1 * minor_entry(mat, i, j)

        return cofactor_matrix(matrix).transpose()

    @staticmethod
    def det_bareis(matrix):
        """
        Adapted from original det_bareis function in Sympy 0.7.3.
        cancel() and expand() are removed from function to speed
        up calculations. Maxima will be used to simplify the result

        Original docstring below:

        Compute matrix determinant using Bareis' fraction-free
        algorithm which is an extension of the well known Gaussian
        elimination method. This approach is best suited for dense
        symbolic matrices and will result in a determinant with
        minimal number of fractions. It means that less term
        rewriting is needed on resulting formulae.
        """
        mat = matrix
        if not mat.is_square:
            raise NonSquareMatrixError()

        m, n = mat[:, :], mat.rows

        if n == 1:
            det = m[0, 0]
        elif n == 2:
            det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        else:
            sign = 1 # track current sign in case of column swap

            for k in range(n - 1):
                # look for a pivot in the current column
                # and assume det == 0 if none is found
                if m[k, k] == 0:
                    for i in range(k + 1, n):
                        if m[i, k] != 0:
                            m.row_swap(i, k)
                            sign *= -1
                            break
                    else:
                        return S.Zero

                # proceed with Bareis' fraction-free (FF)
                # form of Gaussian elimination algorithm
                for i in range(k + 1, n):
                    for j in range(k + 1, n):
                        d = m[k, k] * m[i, j] - m[i, k] * m[k, j]

                        if k > 0:
                            d /= m[k - 1, k - 1]

                        m[i, j] = d

            det = sign * m[n - 1, n - 1]

        return det

    @staticmethod
    def invert(matrix,path_to):
        """
        Returns the numerators of the inverted martix separately from the
        common denominator (the determinant of the matrix)
        """
        common_denom = SymcaToolBox.det_bareis(matrix)
        adjugate = SymcaToolBox.adjugate_matrix(matrix)

        common_denom = SymcaToolBox.maxima_factor(common_denom, path_to)
        #adjugate     = self._maxima_factor('/home/carl/test.txt',adjugate)


        cc_i_sol = adjugate, common_denom
        return cc_i_sol

    @staticmethod
    def maxima_factor(expression,path_to):
        """
        This function is equivalent to the sympy.cancel()
        function but uses maxima instead
        """

        maxima_in_file = path_to + 'in.txt'
        maxima_out_file = path_to + 'out.txt'
        if expression.is_Matrix:
            expr_mat = expression[:, :]
            #print expr_mat
            print 'Simplifying matrix with ' + str(len(expr_mat)) + ' elements'
            for i, e in enumerate(expr_mat):

                sys.stdout.write('*')
                sys.stdout.flush()
                if (i + 1) % 50 == 0:
                    sys.stdout.write(' ' + str(i + 1) + '\n')
                    sys.stdout.flush()
                #print e
                expr_mat[i] = SymcaToolBox.maxima_factor(e,path_to)
            sys.stdout.write('\n')
            sys.stdout.flush()
            return expr_mat
        else:
            batch_string = (
                'stardisp:true;stringout("'
                + maxima_out_file + '",factor(' + str(expression) + '));')
            #print batch_string
            with open(maxima_in_file, 'w') as f:
                f.write(batch_string)

            maxima_command = ['maxima', '--batch=' + maxima_in_file]

            dn = open(devnull, 'w')
            subprocess.call(maxima_command, stdin=dn, stdout=dn, stderr=dn)
            simplified_expression = ''

            with open(maxima_out_file) as f:
                for line in f:
                    if line != '\n':
                        simplified_expression = line[:-2]
            frac = fraction(sympify(simplified_expression))
            #print frac[0].expand()/frac[1].expand()
            return frac[0].expand() / frac[1].expand()

    @staticmethod
    def solve_dep(cc_i_num, scaledk0, scaledl0, num_ind_fluxes,path_to):
        """
        Calculates the dependent control matrices from the independent control
        matrix CC_i_solution
        """

        j_cci_sol = cc_i_num[:num_ind_fluxes, :]
        s_cci_sol = cc_i_num[num_ind_fluxes:, :]

        j_ccd_sol = scaledk0 * j_cci_sol
        s_ccd_sol = scaledl0 * s_cci_sol

        tempmatrix = j_cci_sol
        for matrix in [j_ccd_sol, s_cci_sol, s_ccd_sol]:
            if len(matrix) != 0:
                tempmatrix = tempmatrix.col_join(matrix)

        cc_sol = tempmatrix

        cc_sol = SymcaToolBox.maxima_factor(cc_sol,path_to)

        #print len(j_cci_sol)
        #print len(j_ccd_sol)
        #print len(s_cci_sol)
        #print len(s_ccd_sol)

        return cc_sol

    @staticmethod
    def build_cc_matrix(j, jind, sind, jdep, sdep):
        """
        Produces the matrices j_cci, j_ccd, s_cci and s_ccd
        which holds the symbols for the independent and dependent flux control
        coefficients and the independent and dependent species control
        coefficients respectively
        """
        j_cci = []
        j_ccd = []
        s_cci = []
        s_ccd = []

        for Ji in jind:
            row = []
            for R in j:
                row.append('ccJ' + str(Ji)[2:] + '_' + str(R)[2:])
            j_cci.append(row)

        for Si in sind:
            row = []
            for R in j:
                row.append('cc' + str(Si) + '_' + str(R)[2:])
            s_cci.append(row)

        for Jd in jdep:
            row = []
            for R in j:
                row.append('ccJ' + str(Jd)[2:] + '_' + str(R)[2:])
            j_ccd.append(row)

        for Sd in sdep:
            row = []
            for R in j:
                row.append('cc' + str(Sd) + '_' + str(R)[2:])
            s_ccd.append(row)

        j_cci = Matrix(j_cci)
        j_ccd = Matrix(j_ccd)
        s_cci = Matrix(s_cci)
        s_ccd = Matrix(s_ccd)

        #cc_i = j_cci.col_join(s_cci)
        tempmatrix = j_cci
        for matrix in [j_ccd, s_cci, s_ccd]:
            if len(matrix) != 0:
                tempmatrix = tempmatrix.col_join(matrix)
        cc = tempmatrix

        #print len(j_cci)
        #print len(j_ccd)
        #print len(s_cci)
        #print len(s_ccd)

        return cc

    @staticmethod
    def get_fix_denom(lmatrix, species_independent, species_dependent):
        num_inds = len(species_independent)
        num_deps = len(species_dependent)
        if num_deps == 0:
            return sympify('1')
        else:
            dependent_ls = lmatrix[num_inds:, :]
            denom = sympify('1')
            for row in range(dependent_ls.rows):
                for each in dependent_ls[row, :] * species_independent * -1:
                    denom = denom * each.atoms(Symbol).pop()
                denom = denom * species_dependent[row]
            return denom.nsimplify()

    @staticmethod
    def fix_expressions(cc_num, common_denom_expr, lmatrix, species_independent, species_dependent):

        fix_denom = SymcaToolBox.get_fix_denom(
            lmatrix,
            species_independent,
            species_dependent
        )
        #print fix_denom


        cd_num, cd_denom = fraction(common_denom_expr)

        new_cc_num = cc_num[:, :]
        #print type(new_cc_num)
        for i, each in enumerate(new_cc_num):
            new_cc_num[i] = ((each * cd_denom) / fix_denom).expand()

        return new_cc_num, (cd_num / fix_denom).expand()


    @staticmethod
    def spawn_cc_objects(mod, cc_sol, cc_names, common_denom_expr):

        common_denom = CCBase(
            mod,
            'common_denominator',
            common_denom_expr
        )

        cc_object_list = [common_denom]

        for i, each in enumerate(cc_names):
            cc_object_list.append(
                CCoef(
                    mod,
                    str(each),
                    cc_sol[i],
                    common_denom
                )
            )

        return cc_object_list

    # @staticmethod
    # def expression_to_latex(expression):
    #     #At the moment this function can turn (some) expressions containing
    #     #elasticities and control coefficients into 
    #     #latex strings. One problem is that I assumed that expressions with 
    #     #fractions will always have the form 
    #     #(x1/y1+x2/y2+x3/y3)/(z1/u1+z2/u2+z3/u3). However when the numerator
    #     #only has one term the form is: x1/(y1*(z1/u1+z2/u2+z3/u3))
    #     #and in this case the function does not work correctly. 
    #     if type(expression) != str:
    #         expression = str(expression)

    #     #elasticities
    #     expr = sub(r'ec(\S*?)_(\S*?\b)',r'\\varepsilon^{\1}_{\2}',expression)
    #     #fluxes
    #     expr = sub(r'J_(\S*?\b)',r'J_{\1}',expr)
    #     #controls
    #     expr = sub(r'cc(\S*?)_(\S*?\b)',r'C^{\1}_{\2}',expr)
    #     #main fraction division        
    #     expr = sub(r'\)/\(',r'   ',expr)      
    #     #remove ( and )
    #     expr = sub(r'\)',r'',expr)
    #     expr = sub(r'\(',r'',expr)
    #     #main fraction
    #     expr = sub(r'(\S*[^\)])/([^\(]\S*)',r'\\frac{\1}{\2}',expr)
    #     #sub fractions
    #     expr = sub(r'(.*})\s\s\s(\\frac{.*)',r'\\frac{\1}{\2}',expr)
    #     #times
    #     expr = sub(r'\*',r'\\cdot',expr)


    #     return expr

    

            
