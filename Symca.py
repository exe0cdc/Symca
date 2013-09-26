from sympy.matrices import Matrix

from SymcaToolBox import SymcaToolBox


class Symca(object):
    def __init__(self, mod):
        super(Symca, self).__init__()
        self.mod = mod
        self.tools = SymcaToolBox(
            out_file_name='/home/carl/temp.txt',
            in_file_name='/home/carl/temp1.txt'
        )

        self._nmatrix = None
        self._species = None
        self._num_ind_species = None
        self._species_independent = None
        self._species_dependent = None
        self._fluxes = None
        self._num_ind_fluxes = None
        self._fluxes_independent = None
        self._fluxes_dependent = None
        self._kmatrix = None
        self._lmatrix = None
        self._subs_fluxes = None
        self._scaled_k = None
        self._scaled_l = None
        self._scaled_k0 = None
        self._scaled_l0 = None
        self._es_matrix = None
        self._esL = None
        self._ematrix = None


    @property
    def nmatrix(self):
        if not self._nmatrix:
            self._nmatrix = self.tools.get_nmatrix(self.mod)

        return self._nmatrix

    @property
    def num_ind_species(self):
        if not self._num_ind_species:
            self._num_ind_species = self.tools.get_num_ind_species(self.mod)

        return self._num_ind_species

    @property
    def species(self):
        if not self._species:
            self._species = self.tools.get_species_vector(self.mod)

        return self._species

    @property
    def species_independent(self):
        if not self._species_independent:
            self._species_independent = Matrix(
                self.species[:self.num_ind_species]
            )

        return self._species_independent

    @property
    def species_dependent(self):
        if not self._species_dependent:
            self._species_dependent = Matrix(
                self.species[self.num_ind_species:]
            )

        return self._species_dependent

    @property
    def num_ind_fluxes(self):
        if not self._num_ind_fluxes:
            self._num_ind_fluxes = self.tools.get_num_ind_fluxes(self.mod)

        return self._num_ind_fluxes

    @property
    def fluxes(self):
        if not self._fluxes:
            self._fluxes = self.tools.get_fluxes_vector(self.mod)

        return self._fluxes

    @property
    def fluxes_independent(self):
        if not self._fluxes_independent:
            self._fluxes_independent = Matrix(
                self.fluxes[:self.num_ind_fluxes]
            )

        return self._fluxes_independent

    @property
    def fluxes_dependent(self):
        if not self._fluxes_dependent:
            self._fluxes_dependent = Matrix(
                self.fluxes[self.num_ind_fluxes:]
            )

        return self._fluxes_dependent

    @property
    def kmatrix(self):
        if not self._kmatrix:
            self._kmatrix = Matrix(self.mod.kmatrix)

        return self._kmatrix

    @property
    def lmatrix(self):
        if not self._lmatrix:
            self._lmatrix = Matrix(self.mod.lmatrix)

        return self._lmatrix

    @property
    def subs_fluxes(self):
        if not self._subs_fluxes:
            self._subs_fluxes = self.tools.substitute_fluxes(
                self.fluxes,
                self.kmatrix
            )

        return self._subs_fluxes

    @property
    def scaled_l(self):
        if not self._scaled_l:
            self._scaled_l = self.tools.scale_matrix(
                self.species,
                self.lmatrix,
                self.species_independent
            )

        return self._scaled_l

    @property
    def scaled_k(self):
        if not self._scaled_k:
            self._scaled_k = self.tools.scale_matrix(
                self.subs_fluxes,
                self.kmatrix,
                self.fluxes_independent
            )
        return self._scaled_k


    @property
    def scaled_l0(self):
        if not self._scaled_l0:
            self._scaled_l0 = self.scaled_l[self.num_ind_species:, :]

        return self._scaled_l0


    @property
    def scaled_k0(self):
        if not self._scaled_k0:
            self._scaled_k0 = self.scaled_k[self.num_ind_fluxes:, :]

        return self._scaled_k0

    @property
    def es_matrix(self):
        if not self._es_matrix:
            self._es_matrix = self.tools.get_es_matrix(
                self.mod,
                self.nmatrix,
                self.fluxes,
                self.species
            )

        return self._es_matrix

    @property
    def esL(self):
        if not self._esL:
            self._esL = self.es_matrix * self.scaled_l

        return self._esL

    @property
    def ematrix(self):
        if not self._ematrix:
            self._ematrix = self.tools.simplify_matrix(
                self.scaled_k.row_join(
                    self.esL
                )
            )

        return self._ematrix

    def do_symca(self):

        CC_i_num, common_denom_expr = self.tools.invert(self.ematrix)

        cc_sol = self.tools.solve_dep(
            CC_i_num,
            self.scaled_k0,
            self.scaled_l0,
            self.num_ind_fluxes
        )

        cc_sol, common_denom_expr = self.tools.fix_expressions(
            cc_sol,
            common_denom_expr,
            self.lmatrix,
            self.species_independent,
            self.species_dependent
        )

        cc_names = self.tools.build_cc_matrix(
            self.fluxes,
            self.fluxes_independent,
            self.species_independent,
            self.fluxes_dependent,
            self.species_dependent
        )

        cc_objects = self.tools.spawn_cc_objects(
            self.mod,
            cc_sol,
            cc_names,
            common_denom_expr
        )

        self.CC = cc_objects
        for cc in cc_objects:
            setattr(self, cc.name, cc)

