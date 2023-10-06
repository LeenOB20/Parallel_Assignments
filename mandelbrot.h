#ifndef _MO_MANDELBROT_H
#define _MO_MANDELBROT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <mpi.h>

#define MO_N           2                    /* size of problem space (x, y from -MO_N to MO_N) */
#define MO_THRESHOLD   4                    /* divergence threshold (usually 4) */
#define MO_SIZE        1024                 /* height & width of the canvas in pixels */
#define MO_FILENAME    "./mandelbrot.bmp"   /* default filename of resulting bitmap */
#define MO_MAXITER     2000                 /* default maximum iterations */
#define MO_BLOCKSIZE   1                    /* default # of blocks (rows) to be assigned for 
                                               computation to a slave each time */
#define MO_COLORMIN    0x000000             /* default min color (0x000000 = black) */
#define MO_COLORMAX    0xffffff            /* default max color (0xffffff = white) */
#define MO_COLORMASK   0xffffff            /* default color mask (all tones) */
#define MO_PROGRESS    0                    /* show (1) or hide (0) progress  */
#define MO_PWIDTH      50                   /* progress bar width */
#define MO_PUPDATE     20                   /* update progress bar MO_UPDATE times */


#define MO_CALC        1                    /* "calc" message flag (master to slave) */
#define MO_DATA        2                    /* "data" message flag (slave to master) */
#define MO_STOP        3                    /* "stop" message flag (master to slave) */


#define eprintf(...) { \
    fprintf(stderr, "Error: "); \
    fprintf(stderr, __VA_ARGS__); \
}

#define finalize_exit(error_code) { \
    MPI_Finalize(); \
    exit((error_code)); \
}


typedef struct _mo_complex
{
    double re;                  /* real part */
    double im;                  /* imaginary part */
} mo_complex_t;


typedef struct _mo_opts
{
    int width, height;          /* image width/height */
    char *filename;             /* filename of resulting bitmap */
    double max_re, min_re;      /* real value range */
    double max_im, min_im;      /* imaginary value range */
    int max_iterations;         /* iteration maximum */
    int blocksize;              /* # of rows to be send to slave at once */
    long min_color, max_color;  /* color ranges */
    long color_mask;            /* color mask */
    int show_progress;          /* if 1, show progress */
} mo_opts_t;


typedef struct _mo_scale
{
    double color, im, re;       /* scalings of color, imaginary and 
                                   real part for computation */
} mo_scale_t;


typedef struct _mo_bmp_header
{
    char type[2];               /* "BM" */
    int fsize;                  /* Size of file in bytes */
    int reserved;               /* set to 0 */
    int offbits;                /* Byte offset to actual bitmap data (= 54) */
    int hsize;                  /* Size of BITMAPINFOHEADER, in bytes (= 40) */
    int width;                  /* Width of image, in pixels */
    int height;                 /* Height of images, in pixels */
    short planes;               /* Number of planes in target device (set to 1) */
    short bit_count;            /* Bits per pixel (24 in this case) */
    int compression;            /* Type of compression (0 if no compression) */
    int size_image;             /* Image size, in bytes (0 if no compression) */
    int x_pels_per_meter;       /* Resolution in pixels/meter of display device */
    int y_pels_per_meter;       /* Resolution in pixels/meter of display device */
    int clr_used;               /* Number of colors in the color table (if 0, use 
                                   maximum allowed by bit_count) */
    int clr_important;          /* Number of important colors. If 0, all colors 
                                   are important */
} mo_bmp_header_t;


static int parse_args(int, char **, mo_opts_t *, int, int);
static void print_params(mo_opts_t *, double, double, double);
static void print_usage(char **);
static int master_proc(int, mo_opts_t *);
static int slave_proc(int, mo_opts_t *);
static long mandelbrot(int, int, mo_scale_t *, mo_opts_t *);
static inline void print_progress(int, int);
static int write_bitmap(const char *, int, int, char *);

#endif /* _MO_MANDELBROT_H */
