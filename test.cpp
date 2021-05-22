#include <bits/stdc++.h>
using namespace std;

int main(){
    char output_path[100];
    const char *output_dir = "aiu";
    const char *output_image_name = "hogehoge";
    sprintf(output_path, "%s/%s",output_dir,output_image_name);
    printf("%s\n",output_path);
    return 0;
}