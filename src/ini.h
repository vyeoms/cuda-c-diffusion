#ifndef INI_H
#define INI_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char key[64];
    char val[128];
} IniEntry;

typedef struct {
    IniEntry entries[128];
    int count;
} Ini;

static inline char* ini_strip(char* s)
{
    while (isspace((unsigned char)*s))
        ++s;
    char* end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end))
        *end-- = '\0';
    return s;
}

static inline int ini_load(Ini* ini, const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f)
        return -1;
    ini->count = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char* s = ini_strip(line);
        if (*s == '\0' || *s == '#' || *s == ';' || *s == '[')
            continue;
        char* eq = strchr(s, '=');
        if (!eq)
            continue;
        *eq = '\0';
        char* k = ini_strip(s);
        char* v = ini_strip(eq + 1);
        IniEntry* e = &ini->entries[ini->count++];
        snprintf(e->key, sizeof(e->key), "%s", k);
        snprintf(e->val, sizeof(e->val), "%s", v);
    }
    fclose(f);
    return 0;
}

static inline const char* ini_get(const Ini* ini, const char* key)
{
    for (int i = 0; i < ini->count; ++i)
        if (strcmp(ini->entries[i].key, key) == 0)
            return ini->entries[i].val;
    return NULL;
}

static inline int ini_int(const Ini* ini, const char* key, int def)
{
    const char* v = ini_get(ini, key);
    return v ? atoi(v) : def;
}

static inline float ini_float(const Ini* ini, const char* key, float def)
{
    const char* v = ini_get(ini, key);
    return v ? (float)atof(v) : def;
}

static inline const char* ini_str(const Ini* ini, const char* key, const char* def)
{
    const char* v = ini_get(ini, key);
    return v ? v : def;
}

/* Parse a comma-separated list of ints like "32,64,128" into arr[].
   Returns number of values parsed. */
static inline int ini_int_list(const Ini* ini, const char* key, int* arr, int max_n)
{
    const char* v = ini_get(ini, key);
    if (!v)
        return 0;
    char buf[128];
    strncpy(buf, v, 127);
    buf[127] = '\0';
    int n = 0;
    char* tok = strtok(buf, ",");
    while (tok && n < max_n) {
        arr[n++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    return n;
}

#endif /* INI_H */
