# Keep only the physics columns of a CLASS background table, dropping the per-q
# PSD block.  Dependency-free (awk is everywhere), and it reads the header rather
# than being told column numbers, because those move with dr_N_q.
#
# WHY.  The background table carries lnf_dncdm1[i], dlnfdlnq_*[i],
# f_dr_*_l[j] and f_dr_*_phi[j] for EVERY momentum bin, so it is roughly
# 2*dr_N_q + 3*momenta_bins columns wide: 3300 columns and 383 MB at
# dr_N_q = 1601, and of order a GB at 6401.  Everything the convergence
# diagnostics need (Ecomov, N_eff, a_half, a_ext) is in about 20 of them.
#
#   awk -f slim_background.awk in_background.dat > out_bgslim.dat
#
# The dropped columns are the per-momentum distribution functions.  If you want
# those, keep the gzipped original -- submit_bg.sh does by default.

BEGIN { keep_n = 0 }

# CLASS writes `#  1:z   2:proper time [Gyr]  3:H [1/Mpc] ...`; the index
# PRECEDES its name, and names contain spaces, brackets and dots, so anchor on
# the integers.
/^#/ && /[0-9]+:/ && !have_header {
    line = $0
    while (match(line, /[0-9]+:/)) {
        idx  = substr(line, RSTART, RLENGTH - 1) + 0
        line = substr(line, RSTART + RLENGTH)
        # name runs to the next "<int>:" or to end of line
        rest = line
        if (match(rest, / +[0-9]+:/)) name = substr(rest, 1, RSTART - 1)
        else                          name = rest
        gsub(/^ +| +$/, "", name)
        # keep: geometry/time, every background density and pressure, growth --
        # but NOT the per-bin distribution functions
        if (name !~ /^(lnf_|dlnfdlnq|f_dr_|dlnf)/) {
            keep_n++; keep[keep_n] = idx; kname[keep_n] = name
        }
    }
    have_header = 1
    out = "#"
    for (i = 1; i <= keep_n; i++) out = out sprintf(" %d:%s", i, kname[i])
    print out
    next
}
/^#/ { next }
{
    line = ""
    for (i = 1; i <= keep_n; i++) line = line $(keep[i]) (i < keep_n ? " " : "")
    print line
}
