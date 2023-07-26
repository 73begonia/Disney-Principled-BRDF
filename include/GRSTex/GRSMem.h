#pragma once

//===================================================================================
//��C �ڴ����⺯�� �궨�� ֱ�ӵ�����ӦWinAPI ���Ч��
#ifndef GRS_ALLOC
#define GRS_ALLOC(sz)		HeapAlloc(GetProcessHeap(),0,sz)
#endif // !GRS_ALLOC

#ifndef GRS_CALLOC
#define GRS_CALLOC(sz)		HeapAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,sz)
#endif // !GRS_CALLOC

#ifndef GRS_REALLOC
#define GRS_REALLOC(p,sz)	HeapReAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,p,sz)
#endif // !GRS_REALLOC

#ifndef GRS_FREE
#define GRS_FREE(p)			HeapFree(GetProcessHeap(),0,p)
#endif // !GRS_FREE

#ifndef GRS_MSIZE
#define GRS_MSIZE(p)		HeapSize(GetProcessHeap(),0,p)
#endif // !GRS_MSIZE

#ifndef GRS_MVALID
#define GRS_MVALID(p)		HeapValidate(GetProcessHeap(),0,p)
#endif // !GRS_MVALID

#ifndef GRS_SAFE_FREE
//����������ʽ���ܻ���������,p��������ʹ��ָ�����,�����Ǳ��ʽ
#define GRS_SAFE_FREE(p) if( nullptr != (p) ){ HeapFree(GetProcessHeap(),0,(p));(p)=nullptr; }
#endif // !GRS_SAFEFREE

#ifndef GRS_SAFE_DELETE
//����������ʽ���ܻ���������,p��������ʹ��ָ�����,�����Ǳ��ʽ
#define GRS_SAFE_DELETE(p) if( nullptr != (p) ){ delete p; (p) = nullptr; }
#endif // !GRS_SAFEFREE

#ifndef GRS_SAFE_DELETE_ARRAY
//����������ʽ���ܻ���������,p��������ʹ��ָ�����,�����Ǳ��ʽ
#define GRS_SAFE_DELETE_ARRAY(p) if( nullptr != (p) ){ delete[] p; (p) = nullptr; }
#endif // !GRS_SAFEFREE


#ifndef GRS_OPEN_HEAP_LFH
//������������ڴ򿪶ѵ�LFH����,���������
#define GRS_OPEN_HEAP_LFH(h) \
    ULONG  ulLFH = 2;\
    HeapSetInformation((h),HeapCompatibilityInformation,&ulLFH ,sizeof(ULONG) ) ;
#endif // !GRS_OPEN_HEAP_LFH
//===================================================================================