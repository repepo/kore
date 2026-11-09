#!/usr/bin/env python3
'''
Writes the two tables of docs/compatibility.md, the grid of the pairs of features and the table of the forced runs, from
the cells below. Use as

    python3 tools/compat_tables.py           (rewrites the tables in docs/compatibility.md)
    python3 tools/compat_tables.py --check   (exit code 1 if the page differs from the cells)

NEW (2026-11-09): each cell is a mark, the notes it runs under and the runs that check it:
  ✓  runs, and the runs named check it (shown when the pointer is on the mark);
  ○  runs, and no suite checks it;
  ✗  stops at startup;
  –  has no meaning (two values of one parameter, an inner core in a full sphere, a viscosity without a flow).
A cell is ✓ only where a run with both features compares its solution with theory, an independent computation, the
conductive_IC branch or a budget that closes (spin_doctor), not where it only compares a run with an earlier one
(regress.sh) or only starts or stops. The notes are the items '- **1.**' (grid) and '- **a.**' (forcings) of the page,
which must hold every note the cells use and no other. A new feature or a new stop: its cells here, its notes on the
page, then run this script.
'''
import html, os, re, sys

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'compatibility.md')
Y, O, X, N = '✓', '○', '✗', '–'

# --- the grid: one cell for each pair of features (the upper triangle, in the order of CODES)
CODES = 'FS Inv H0 S C DR U Cf Sph B cIC Bes TWA Man Rot Kd Kt'.split()
G = {}
def c(a, b, mark, notes=(), tests=''):
    assert (a, b) not in G and (b, a) not in G, (a, b)
    assert CODES.index(a) < CODES.index(b), (a, b)
    assert (mark == Y) == bool(tests), (a, b, mark, tests)
    G[(a, b)] = (mark, tuple(notes), tests)

# full sphere
c('FS', 'Inv', Y, (), 'centrifugal: spin-over, spheroid_check; magnetic: rcf_inv_s2; bgflow: frame_sphere_inviscid; cic_forcings: lib_inviscid')
c('FS', 'H0', Y, (), 'decay: full_sphere; mantle: decay_sphere; twa: decay_sphere_p3')
c('FS', 'S', Y, (1,), 'ci: test_spinover; heat_dr: sun; bgflow: tor_sphere; composition: th_sphere; tidal: f14_sphere_wu')
c('FS', 'C', Y, (1,), 'composition: co_sphere, co_sphere_flux, co_sphere_nodiff; bgflow: frame_sphere_inviscid')
c('FS', 'DR', Y, (), 'ci: test_spinover; heat_dr: sun; bgflow: dr_sphere; magnetic: rg_fs, bdr_fs')
c('FS', 'U', Y, (), 'bgflow: frame_sphere, tor_sphere, sphere, inviscid')
c('FS', 'Cf', Y, (2,), 'centrifugal: spin-over, ru_bq_fs1, bud_bq_fs, bm_fs, spheroid_modes')
c('FS', 'Sph', Y, (2,), 'centrifugal: spheroid_check, bud_sph_fs, bms_fs')
c('FS', 'B', Y, (3,), 'decay: full_sphere; magnetic: rcf_ax_m1, bf_s2_m1, rg_fs; couette: rigid_sphere; sheet: lz_sphere; bgflow: sphere')
c('FS', 'cIC', N)
c('FS', 'Bes', N)
c('FS', 'TWA', Y, (4,), 'twa: decay_sphere_p3, cic_sphere_slip; forced_field: fs_f9_twa')
c('FS', 'Man', Y, (), 'mantle: decay_sphere, cic_sphere_slip; forced_field: def_fs_man')
c('FS', 'Rot', X)
c('FS', 'Kd', Y, (), 'drag: k_sphere')
c('FS', 'Kt', Y, (), 'drag: t_sphere')
# inviscid fluid (a full sphere: background.check stops an inviscid shell)
c('Inv', 'H0', N)
c('Inv', 'S', Y, (1,), 'centrifugal: monopole, ideal, shape_check; bgflow: frame_sphere_inviscid')
c('Inv', 'C', Y, (1,), 'bgflow: frame_sphere_inviscid')
c('Inv', 'DR', O)
c('Inv', 'U', Y, (), 'bgflow: frame_sphere_inviscid, inviscid')
c('Inv', 'Cf', Y, (), 'centrifugal: spin-over, relabelling, clairaut, ideal, spheroid_modes, shape_check')
c('Inv', 'Sph', Y, (), 'centrifugal: spheroid_check')
c('Inv', 'B', Y, (3, 5), 'magnetic: rcf_inv_s2; bgflow: frame_sphere_inviscid')
c('Inv', 'cIC', X)
c('Inv', 'Bes', X)
c('Inv', 'TWA', O, (4, 5))
c('Inv', 'Man', O)
c('Inv', 'Rot', X)
c('Inv', 'Kd', X)
c('Inv', 'Kt', X)
# no flow
c('H0', 'S', Y, (), 'decay: entropy, entropy_flux; composition: th_decay')
c('H0', 'C', Y, (), 'composition: co_decay')
c('H0', 'DR', Y, (), 'decay: diffrot_hydro0')
c('H0', 'U', O)
c('H0', 'Cf', O)
c('H0', 'Sph', O)
c('H0', 'B', Y, (), 'decay: insulating, full_sphere; twa: decay_cmb_*; mantle: decay_uniform, decay_exp')
c('H0', 'cIC', Y, (), 'decay: conducting_ic; mantle: decay_cic')
c('H0', 'Bes', X)
c('H0', 'TWA', Y, (), 'twa: decay_cmb_*, decay_sphere_p3')
c('H0', 'Man', Y, (), 'mantle: decay_uniform, decay_exp, decay_sphere, decay_cic')
c('H0', 'Rot', X)
c('H0', 'Kd', X)
c('H0', 'Kt', X)
# entropy
c('S', 'C', Y, (), 'composition: cic_two_zone, both, wu_both; bgflow: wu_all')
c('S', 'DR', Y, (6,), 'heat_dr: sun, rfix1, s96; ci: test_spinover; composition: th_diff_rot; tidal: f14_m1_diffrot; bgflow: dr_shell')
c('S', 'U', Y, (), 'bgflow: tor_shell, frame_adiabatic, wu_all')
c('S', 'Cf', Y, (), 'centrifugal: ru_wu_m1, bud_wu_cl, ideal, shape_check; composition: th_deformed')
c('S', 'Sph', Y, (), 'centrifugal: ru_sph_m1, bud_sph_ns, spheroid_check; composition: th_spheroid')
c('S', 'B', Y, (), 'sheet: wu_tor_out; composition: th_field; inner_core: slip_wu; forced_field: wu_f9_bes; couette: wu_field')
c('S', 'cIC', Y, (), 'composition: th_field; bgflow: wu_all; inner_core: slip_wu, f6_wu; couette: wu_field')
c('S', 'Bes', Y, (), 'forced_field: wu_f9_bes, wu_lib_man')
c('S', 'TWA', Y, (), 'composition: th_solids_field; forced_field: wu_f14_twa; couette: wu_field')
c('S', 'Man', Y, (), 'composition: th_field; bgflow: wu_all; forced_field: wu_lib_man')
c('S', 'Rot', Y, (), 'composition: th_solids, th_solids_field')
c('S', 'Kd', Y, (), 'drag: forced13; couette: f13_drag')
c('S', 'Kt', Y, (), 'drag: th, th2; composition: th_drag')
# composition
c('C', 'DR', Y, (), 'composition: co_diff_rot, wu_both; bgflow: dr_shell')
c('C', 'U', Y, (), 'bgflow: tor_shell, frame_adiabatic, wu_all')
c('C', 'Cf', Y, (), 'composition: co_deformed')
c('C', 'Sph', Y, (), 'composition: co_spheroid')
c('C', 'B', Y, (), 'composition: co_field, co_solids_field; bgflow: wu_all')
c('C', 'cIC', Y, (), 'composition: co_field; bgflow: wu_all')
c('C', 'Bes', O)
c('C', 'TWA', Y, (), 'composition: co_solids_field')
c('C', 'Man', Y, (), 'composition: co_field; bgflow: wu_all')
c('C', 'Rot', Y, (), 'composition: co_solids, co_solids_field')
c('C', 'Kd', O)
c('C', 'Kt', Y, (), 'composition: co_drag')
# differential rotation
c('DR', 'U', O)
c('DR', 'Cf', Y, (), 'centrifugal: ru_wu_dr, bud_wu_dr, bm_gen')
c('DR', 'Sph', X)
c('DR', 'B', Y, (7,), 'decay: diffrot_hydro1; magnetic: rg_axial, bdr_pol, bgen, bfer, bfdr; bgflow: dr_shell')
c('DR', 'cIC', Y, (8,), 'inner_core: slip_offset, f10_diffrot_cheb; magnetic: rg_cic; mantle: rigid_offset; couette: diffrot')
c('DR', 'Bes', Y, (8,), 'inner_core: f10_diffrot_bes; forced_field: bessel_chebys_dr')
c('DR', 'TWA', Y, (8,), 'twa: rigid_offset, sheared_walls')
c('DR', 'Man', Y, (8,), 'mantle: rigid_offset, sheared; forced_field: def_dr')
c('DR', 'Rot', X)
c('DR', 'Kd', Y, (), 'drag: k_rigid')
c('DR', 'Kt', X)
# background flow
c('U', 'Cf', X)
c('U', 'Sph', X)
c('U', 'B', Y, (7,), 'bgflow: tor_shell, frame_field_solids, wu_all, sphere, twa, f12_field')
c('U', 'cIC', Y, (8,), 'bgflow: frame_field_solids, wu_all, f12_field')
c('U', 'Bes', O, (8,))
c('U', 'TWA', Y, (8,), 'bgflow: frame_field_twa, twa, f8_twa')
c('U', 'Man', Y, (8,), 'bgflow: frame_field_solids, wu_all')
c('U', 'Rot', X)
c('U', 'Kd', O)
c('U', 'Kt', X)
# deformed body
c('Cf', 'Sph', N)
c('Cf', 'B', Y, (9,), 'centrifugal: bm_uniform, bm_tor, eig_check (with a field), spheroid_check (first order)')
c('Cf', 'cIC', Y, (10,), 'centrifugal: bm_cic, eig_check (decay with a core), spheroid_check (first order), fcn.sh')
for b in ('Bes', 'TWA', 'Man', 'Rot'):
    c('Cf', b, X)
c('Cf', 'Kd', Y, (), 'drag: deformed_check (axial, first, relabelled)')   # NEW (2026-11-09)
c('Cf', 'Kt', Y, (), 'drag: deformed_check (first, relabelled)')
c('Sph', 'B', Y, (9,), 'centrifugal: bms_ins, bms_tor, spheroid_check')
c('Sph', 'cIC', Y, (10,), 'centrifugal: bms_cic, spheroid_check')
c('Sph', 'Bes', X)
c('Sph', 'TWA', Y, (15,), 'mantle: spheroid_mantle_check (thin layers, f → 0, slip), derive_spheroid; wobble: wobble_field_check')   # NEW (2026-11-09)
c('Sph', 'Man', Y, (), 'mantle: spheroid_mantle_check (a larger spheroid, thin layers, f → 0, slip); wobble: wobble_field_check')
c('Sph', 'Rot', X)
c('Sph', 'Kd', Y, (), 'drag: deformed_check (axial, spin-over, weak, stiff)')   # NEW (2026-11-09)
c('Sph', 'Kt', Y, (), 'drag: deformed_check (spin-over, weak)')
# the field and its solids
c('B', 'cIC', Y, (), 'inner_core: slip_axial_m0, slip_dipole; decay: conducting_ic; magnetic: bud_m1, tb_mal_m1, b_s2_cic')
c('B', 'Bes', Y, (), 'inner_core: dip_libration; forced_field: lib_bes, bessel_chebys_f9; couette: c_bes')
c('B', 'TWA', Y, (), 'inner_core: dip_twa_mantle, dip_twa_core; twa: cic_noslip_both; sheet: pol_layer_out; rotdyn: eq_twa')
c('B', 'Man', Y, (), 'inner_core: dip_resolved_mantle; mantle: cic_noslip, sigma_zero, induction')
c('B', 'Rot', Y, (11,), 'rotdyn: cic_twa, cic_s2, cic_axial, wu_field; inner_core: dip_rotdyn; drag: r_twa_k')
c('B', 'Kd', Y, (), 'drag: r_twa_k, r_res_k; sheet: tor_out_drag')
c('B', 'Kt', Y, (), 'drag: m_twa_t, wu_t_field')
c('cIC', 'Bes', N)
c('cIC', 'TWA', Y, (12,), 'couette: rigid_mu, wu_field; forced_field: wu_f14_twa, f10_twa')
c('cIC', 'Man', Y, (), 'mantle: cic_innercore, decay_cic, rigid_offset; forced_field: lib_man; couette: c_man; rotdyn: cic_s2')
c('cIC', 'Rot', Y, (), 'rotdyn: cic_s2, wu_field; inner_core: dip_rotdyn; drag: r_res_k')
c('cIC', 'Kd', Y, (), 'drag: r_res_k')
c('cIC', 'Kt', Y, (), 'drag: wu_t_field')
c('Bes', 'TWA', Y, (12,), 'cic_forcings: lib7_field')
c('Bes', 'Man', Y, (), 'forced_field: wu_lib_man, f7m2_man')
c('Bes', 'Rot', X)
c('Bes', 'Kd', O, (13,))
c('Bes', 'Kt', O, (13,))
c('TWA', 'Man', Y, (14,), 'rotdyn: cic_axial')
c('TWA', 'Rot', Y, (), 'rotdyn: cic_twa, cic_axial; drag: r_twa_k')
c('TWA', 'Kd', Y, (), 'drag: r_twa_k')
c('TWA', 'Kt', Y, (), 'drag: m_twa_t')
c('Man', 'Rot', Y, (), 'rotdyn: cic_s2, cic_axial; drag: r_res_k')
c('Man', 'Kd', Y, (), 'drag: r_res_k')
c('Man', 'Kt', Y, (), 'drag: wu_t_field')
c('Rot', 'Kd', Y, (), 'drag: r_k, r_twa_k, r_res_k, r_stiff, r_weak')
c('Rot', 'Kt', Y, (), 'drag: r_t, wu_t_field')
c('Kd', 'Kt', X)
_missing = [(a, b) for i, a in enumerate(CODES) for b in CODES[i + 1:] if (a, b) not in G]
assert not _missing, _missing

# --- the forced runs: one cell for each forcing and each feature that a forced run can take
FC = 'FS Inv S C DR U B cIC Bes TWA Man Kd Kt'.split()
F = {}
def f(n, col, mark, notes='', tests=''):
    assert col in FC and col not in F.get(n, {}), (n, col)
    assert (mark == Y) == bool(tests), (n, col)
    F.setdefault(n, {})[col] = (mark, tuple(notes), tests)

# 1, the wave-like tide
f(1, 'FS', Y, 'a', 'cic_forcings: f1_m0_sphere, f1_m1_sphere, f1_m0_rigid, f1_m1_rigid')
f(1, 'Inv', Y, 'a', 'cic_forcings: f1_m1_inviscid')
for col in ('S', 'C', 'B', 'cIC', 'Bes', 'TWA', 'Man', 'Kd', 'Kt'):
    f(1, col, X)
f(1, 'DR', Y, '', 'cic_forcings: f1_m2_dr')
f(1, 'U', Y, '', 'bgflow: f1')
# 2, the eccentricity tide of Enceladus
f(2, 'FS', X); f(2, 'Inv', X)
f(2, 'S', Y, '', 'cic_forcings: f2_wu'); f(2, 'C', O); f(2, 'DR', Y, '', 'cic_forcings: f2_dr'); f(2, 'U', O)
f(2, 'B', Y, '', 'cic_forcings: f2_field'); f(2, 'cIC', X); f(2, 'Bes', X); f(2, 'TWA', O, 'bc'); f(2, 'Man', X)
f(2, 'Kd', X); f(2, 'Kt', X)
# 6, the radial flow at the inner wall
f(6, 'FS', X); f(6, 'Inv', X)
f(6, 'S', Y, '', 'inner_core: f6_wu'); f(6, 'C', O); f(6, 'DR', O); f(6, 'U', O)
f(6, 'B', Y, '', 'inner_core: f6_axial, f6_deforming; forced_field: bessel_chebys_f6, slip_f6_twa, def_f6')
f(6, 'cIC', Y, 'c', 'inner_core: f6_axial_cheb, f6_deforming, f6_wu; forced_field: def_f6')
f(6, 'Bes', Y, 'cf', 'inner_core: f6_axial_bes; forced_field: bessel_chebys_f6')
f(6, 'TWA', Y, 'c', 'forced_field: slip_f6_twa, f6_muw_fixed')
f(6, 'Man', O); f(6, 'Kd', O); f(6, 'Kt', Y, 'i', 'drag: forced')
# 7, the libration of the walls
f(7, 'FS', Y, '', 'cic_forcings: lib7_sphere')
f(7, 'Inv', Y, 'd', 'forcing: rows_check (f7_m2_inviscid)')
f(7, 'S', Y, '', 'cic_forcings: lib7_wu_dr; forced_field: wu_lib_man')
f(7, 'C', Y, '', 'composition: co_libration')
f(7, 'DR', Y, '', 'cic_forcings: lib7_wu_dr')
f(7, 'U', Y, '', 'bgflow: f7')
f(7, 'B', Y, '', 'forced_field: lib_ins, lib_cheb; inner_core: dip_libration, f7_axial_cheb; cic_forcings: lib7_field')
f(7, 'cIC', Y, 'ce', 'forced_field: lib_cheb, lib_man; inner_core: f7_axial_cheb, f7_dipole_cheb')
f(7, 'Bes', Y, 'cef', 'forced_field: lib_bes, lib_s2_bes; inner_core: dip_libration, f7_axial_bes; cic_forcings: lib7_field')
f(7, 'TWA', Y, 'ce', 'forced_field: lib_twa; cic_forcings: lib7_field')
f(7, 'Man', Y, '', 'forced_field: lib_man, wu_lib_man, f7m2_man')
f(7, 'Kd', X); f(7, 'Kt', X)
# 8, the libration in the frame of the mantle
f(8, 'FS', Y, '', 'cic_forcings: lib_sphere, lib_inviscid')
f(8, 'Inv', Y, '', 'cic_forcings: lib_inviscid')
f(8, 'S', Y, '', 'cic_forcings: lib_wu_dr'); f(8, 'C', O); f(8, 'DR', Y, '', 'cic_forcings: lib_wu_dr')
f(8, 'U', Y, '', 'bgflow: f8, f8_twa')
f(8, 'B', Y, '', 'cic_forcings: c_f8_field, lib_field, lib_twa, lib_man')
f(8, 'cIC', Y, 'e', 'cic_forcings: c_f8_field, lib_cheb_slip')
f(8, 'Bes', Y, 'e', 'cic_forcings: lib_field')
f(8, 'TWA', Y, 'e', 'cic_forcings: lib_twa; bgflow: f8_twa')
f(8, 'Man', Y, '', 'cic_forcings: lib_man')
f(8, 'Kd', Y, '', 'cic_forcings: lib_drag'); f(8, 'Kt', O)
# 9, a radial flow of degree 2 at the walls
f(9, 'FS', Y, '', 'forcing: rows_check (f9_sphere, f9_sphere_bci0); forced_field: fs_f9_twa, def_fs_man')
f(9, 'Inv', O)
f(9, 'S', Y, '', 'forced_field: wu_f9_bes, def_wu_f9'); f(9, 'C', O)
f(9, 'DR', Y, '', 'forced_field: dr_bes, dr_cheb, def_dr'); f(9, 'U', O)
f(9, 'B', Y, '', 'forced_field: f9_ins, f9_bes, f9_cheb, malkus_twa, malkus_man')
f(9, 'cIC', Y, 'c', 'forced_field: f9_cheb, f9_man_cheb, def_g21, def_wu_f9')
f(9, 'Bes', Y, 'cf', 'forced_field: f9_bes, wu_f9_bes')
f(9, 'TWA', Y, 'c', 'forced_field: f9_twa, malkus_twa, layer_*_rigid_*, layer_*_deforming_*')
f(9, 'Man', Y, '', 'forced_field: malkus_man, f9_man_cheb, layer_*_rigid_*, layer_*_deforming_*')
f(9, 'Kd', Y, 'g', 'forcing: rows_check (f9_drag); drag: forced9'); f(9, 'Kt', O, 'gi')
# 10, a radial flow of degree l = m at the outer wall
f(10, 'FS', Y, '', 'forcing: rows_check (f10_inviscid)')
f(10, 'Inv', Y, '', 'forcing: rows_check (f10_inviscid)')
f(10, 'S', O); f(10, 'C', O)
f(10, 'DR', Y, '', 'inner_core: f10_diffrot_cheb, f10_diffrot_bes'); f(10, 'U', O)
f(10, 'B', Y, '', 'forced_field: f10_twa, def_f10; inner_core: f10_mu_sigma_cheb')
f(10, 'cIC', Y, '', 'inner_core: f10_mu_sigma_cheb; forced_field: f10_twa')
f(10, 'Bes', Y, '', 'inner_core: f10_mu_sigma_bes, f10_diffrot_bes')
f(10, 'TWA', Y, 'c', 'forced_field: f10_twa')
f(10, 'Man', Y, '', 'forced_field: def_f10')
f(10, 'Kd', X); f(10, 'Kt', X)
# 11, the random Reynolds stresses
f(11, 'FS', Y, '', 'rsf: sp_fs (spectrum_check)')
f(11, 'Inv', X)
f(11, 'S', O); f(11, 'C', O); f(11, 'DR', O); f(11, 'U', O)
f(11, 'B', Y, '', 'rsf: sp_mag (spectrum_check)')
f(11, 'cIC', Y, '', 'rsf: sp_mag (spectrum_check)')
f(11, 'Bes', O, 'h'); f(11, 'TWA', O); f(11, 'Man', O)
f(11, 'Kd', O); f(11, 'Kt', O, 'i')
# 12, the steady rotation of the walls
f(12, 'FS', Y, '', 'couette: rigid_sphere; forcing: rows_check (f12_sphere)')
f(12, 'Inv', X)
f(12, 'S', Y, '', 'couette: wu_thermal, wu_field'); f(12, 'C', O)
f(12, 'DR', Y, '', 'couette: diffrot'); f(12, 'U', Y, '', 'bgflow: f12, f12_field')
f(12, 'B', Y, '', 'couette: c_ins, rigid_shell, wu_field, malkus')
f(12, 'cIC', Y, '', 'couette: c_cheb, c_mu, rigid_shell, rigid_mu')
f(12, 'Bes', Y, '', 'couette: c_bes, c_s2_bes')
f(12, 'TWA', Y, '', 'couette: c_twa, c_muw, rigid_twa, slip_twa')
f(12, 'Man', Y, '', 'couette: c_man, rigid_shell, slip_man')
f(12, 'Kd', Y, 'j', 'couette: slip_drag'); f(12, 'Kt', O, 'ij')
# 13, the entropy at the inner wall
f(13, 'FS', X); f(13, 'Inv', X)
f(13, 'S', Y, '', 'couette: f13; forced_field: wu_f13_cheb'); f(13, 'C', O); f(13, 'DR', O); f(13, 'U', O)
f(13, 'B', Y, '', 'forced_field: wu_f13_cheb'); f(13, 'cIC', Y, '', 'forced_field: wu_f13_cheb')
f(13, 'Bes', O); f(13, 'TWA', O); f(13, 'Man', O)
f(13, 'Kd', Y, '', 'drag: forced13; couette: f13_drag'); f(13, 'Kt', O, 'ik')
# 14, the tide that moves the walls
f(14, 'FS', Y, '', 'tidal: sphere_m2, f14_sphere_wu')
f(14, 'Inv', Y, '', 'forcing: rows_check (f14_inviscid)')
f(14, 'S', Y, '', 'tidal: f14_m2_thermal; forced_field: wu_f14_twa'); f(14, 'C', O)
f(14, 'DR', Y, '', 'tidal: f14_m1_diffrot, f14_m2_wu_conical'); f(14, 'U', O)
f(14, 'B', Y, '', 'forced_field: wu_f14_twa, f14_twa_def, def_wu_f14')
f(14, 'cIC', Y, 'c', 'forced_field: wu_f14_twa, def_wu_f14')
f(14, 'Bes', O, 'cf')
f(14, 'TWA', Y, 'c', 'forced_field: wu_f14_twa, f14_twa_def')
f(14, 'Man', Y, '', 'forced_field: def_wu_f14')
f(14, 'Kd', X); f(14, 'Kt', X)
for _n in F:
    assert set(F[_n]) == set(FC), (_n, set(FC) - set(F[_n]))

NAMES = {1: '1, wave-like tide', 2: '2, eccentricity tide of Enceladus', 6: '6, radial flow at the inner wall',
         7: '7, libration of the walls', 8: '8, libration in the frame of the mantle', 9: '9, radial flow at the walls',
         10: '10, radial flow at the outer wall', 11: '11, random Reynolds stresses', 12: '12, walls that turn steadily',
         13: '13, entropy at the inner wall', 14: '14, tide that moves the walls'}
LINKS = {1: 'tidal-forcing.md#the-wave-like-tide-forcing-1', 2: 'tidal-forcing.md#the-eccentricity-tide-of-enceladus-forcing-2',
         7: 'libration.md#forcing-7-the-walls-librate', 8: 'libration.md#forcing-8-the-frame-of-the-mantle',
         11: 'forcing.md', 12: 'couette-flow.md', 13: 'thermal-forcing.md', 14: 'tidal-forcing.md'}


FEATURES = {'FS': 'a full sphere', 'Inv': 'an inviscid fluid', 'H0': 'no flow', 'S': 'the entropy', 'C': 'the composition',
            'DR': 'a differential rotation', 'U': 'a background flow', 'Cf': 'the centrifugal deformation',
            'Sph': 'the exact spheroid', 'B': 'a magnetic field', 'cIC': 'a conducting inner core',
            'Bes': 'a conducting inner core at one frequency', 'TWA': 'a thin conducting layer',
            'Man': 'a resolved conducting mantle', 'Rot': 'the rotating solids', 'Kd': 'a constant drag',
            'Kt': 'the drag of the topography'}
CLASS = {Y: 'y', O: 'o', X: 'x', N: 'n'}


def code(b):
    return '<abbr title="%s">%s</abbr>' % (FEATURES[b], b)


def cell(mark, notes, tests):
    title = ' title="%s"' % html.escape('tested by tests/' + tests.replace('; ', '; tests/'), quote=True) if tests else ''
    s = '<span class="%s"%s>%s</span>' % (CLASS[mark], title, mark)
    if notes:
        s += '<sup>%s</sup>' % ','.join(str(n) for n in notes)
    return s


def grid():
    out = ['| | ' + ' | '.join(code(b) for b in CODES) + ' |', '| --- ' * (len(CODES) + 1) + '|']
    for a in CODES:
        out.append('| %s | ' % code(a) + ' | '.join('<span class="d"></span>' if a == b else cell(*G.get((a, b), G.get((b, a))))
                                                   for b in CODES) + ' |')
    return '\n'.join(out)


def forcings():
    out = ['| forcing | ' + ' | '.join(code(b) for b in FC) + ' |', '| --- ' * (len(FC) + 1) + '|']
    for n in sorted(F):
        name = '[%s](%s)' % (NAMES[n], LINKS[n]) if n in LINKS else NAMES[n]
        out.append('| %s | ' % name + ' | '.join(cell(*F[n][b]) for b in FC) + ' |')
    return '\n'.join(out)


def fill(text):
    '''the page with its two tables written from the cells; stops if the notes of the page are not those of the cells'''
    for key, table in (('grid', grid()), ('forcings', forcings())):
        start, end = '<!-- compat_tables.py: %s -->' % key, '<!-- end of %s -->' % key
        i, j = text.find(start), text.find(end)
        if i < 0 or j < i:
            sys.exit('tools/compat_tables.py: the page has no markers %r ... %r' % (start, end))
        text = text[:i + len(start)] + '\n\n<div class="compat" markdown="1">\n\n' + table + '\n\n</div>\n\n' + text[j:]
    used = {str(n) for v in G.values() for n in v[1]}, {n for row in F.values() for v in row.values() for n in v[1]}
    listed = set(re.findall(r'^- \*\*(\d+)\.\*\*', text, re.M)), set(re.findall(r'^- \*\*([a-z])\.\*\*', text, re.M))
    for what, u, l in (('grid', used[0], listed[0]), ('forcings', used[1], listed[1])):
        if u != l:
            sys.exit('tools/compat_tables.py: the notes of the %s on the page (%s) are not those of its cells (%s)'
                     % (what, ' '.join(sorted(l, key=lambda s: (len(s), s))), ' '.join(sorted(u, key=lambda s: (len(s), s)))))
    return text


if __name__ == '__main__':
    old = open(PAGE, encoding='utf-8').read()
    new = fill(old)
    if sys.argv[1:] == ['--check']:
        sys.exit(0 if new == old else 'tools/compat_tables.py: docs/compatibility.md differs from the cells: run it')
    if sys.argv[1:]:
        sys.exit(__doc__)
    if new != old:
        open(PAGE, 'w', encoding='utf-8').write(new)
    marks = lambda cells: ', '.join('%s %d' % (m, sum(v[0] == m for v in cells)) for m in (Y, O, X, N))
    print('docs/compatibility.md: grid %s; forcings %s' % (marks(G.values()), marks([v for row in F.values() for v in row.values()])))
